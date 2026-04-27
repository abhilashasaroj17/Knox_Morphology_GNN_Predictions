"""
Knox County City2Graph - Step 2c: Graph Visualizations
=======================================================
Visualizes the actual graphs built in Step 2:
  - OD Graph: nodes = TAZ zones, edges = trip flows
  - Road network graph: Overture segments as a spatial graph

Run with:
    .venv\Scripts\python.exe step2c_graph_viz.py
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
from pathlib import Path

OUT_DIR = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph\outputs")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load data
# =============================================================================
print("Loading data...")
zones    = gpd.read_file(OUT_DIR / "zones_reprojected.gpkg")
od_int   = pd.read_csv(OUT_DIR / "od_long_2026_internal.csv")
od_stats = pd.read_csv(OUT_DIR / "od_zone_stats.csv")
zones    = zones.merge(od_stats, on="zone_id", how="left")
zones["centroid_x"] = zones.geometry.centroid.x
zones["centroid_y"] = zones.geometry.centroid.y
print(f"  Zones: {len(zones):,}  |  OD pairs: {len(od_int):,}")

# =============================================================================
# Build NetworkX OD graph
# =============================================================================
print("Building OD graph...")
G_od = nx.DiGraph()
for _, row in zones.iterrows():
    G_od.add_node(
        int(row["zone_id"]),
        x=row["centroid_x"],
        y=row["centroid_y"],
        production=row.get("production", 0),
        attraction=row.get("attraction", 0),
    )
for _, row in od_int.iterrows():
    G_od.add_edge(int(row["origin"]), int(row["destination"]), weight=float(row["flow"]))

print(f"  OD Graph: {G_od.number_of_nodes()} nodes, {G_od.number_of_edges():,} edges")

# Node positions = zone centroids
pos = {n: (G_od.nodes[n]["x"], G_od.nodes[n]["y"]) for n in G_od.nodes}

# =============================================================================
# Fig G1: OD Graph — full, top-flow edges only (otherwise unreadable)
# =============================================================================
print("Fig G1: OD graph — top flows...")

TOP_N = 2000   # show only the strongest 2000 OD edges
top_edges     = od_int.nlargest(TOP_N, "flow")
G_top         = nx.DiGraph()
G_top.add_nodes_from(G_od.nodes(data=True))
for _, row in top_edges.iterrows():
    G_top.add_edge(int(row["origin"]), int(row["destination"]), weight=float(row["flow"]))

# Node size = log(production), color = net_flow
node_ids   = list(G_top.nodes)
prod_vals  = np.array([G_od.nodes[n].get("production", 1) for n in node_ids])
attr_vals  = np.array([G_od.nodes[n].get("attraction", 1) for n in node_ids])
net_flow   = prod_vals - attr_vals
node_sizes = np.log1p(prod_vals) * 3

# Edge weights for width/alpha
edge_weights = np.array([G_top[u][v]["weight"] for u, v in G_top.edges()])
w_norm       = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-9)

fig, ax = plt.subplots(figsize=(14, 11))

# Draw zone polygons as background
zones.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.3, zorder=1)

# Draw edges (colored by flow magnitude)
edge_colors = cm.YlOrRd(w_norm)
edge_list = list(G_top.edges())
edge_index = {e: i for i, e in enumerate(edge_list)}
for (u, v), color, alpha in zip(G_top.edges(), edge_colors, 0.15 + w_norm * 0.6):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    idx = edge_index[(u, v)]
    ax.plot([x0, x1], [y0, y1], color=color, alpha=float(alpha),
            linewidth=0.4 + w_norm[idx] * 1.2, zorder=2)

# Edge colorbar
edge_flow_min = edge_weights.min()
edge_flow_max = edge_weights.max()
sm_edges = cm.ScalarMappable(cmap="YlOrRd",
    norm=mcolors.Normalize(vmin=edge_flow_min, vmax=edge_flow_max))
sm_edges.set_array([])
cb_edges = plt.colorbar(sm_edges, ax=ax, label="OD Flow (trips)",
                        shrink=0.40, pad=0.01, location="left")
cb_edges.ax.yaxis.set_label_position("left")
cb_edges.ax.yaxis.tick_left()

# Draw nodes (colored by net_flow: red=generator, blue=attractor)
vmax = np.abs(net_flow).quantile(0.95) if hasattr(np.abs(net_flow), "quantile") else np.percentile(np.abs(net_flow), 95)
norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
node_colors = cm.RdBu_r(norm(net_flow))
xs = [pos[n][0] for n in node_ids]
ys = [pos[n][1] for n in node_ids]
sc = ax.scatter(xs, ys, s=node_sizes, c=net_flow, cmap="RdBu_r",
                norm=norm, zorder=3, edgecolors="white", linewidths=0.3, alpha=0.9)
plt.colorbar(sc, ax=ax, label="Net Flow (production - attraction)", shrink=0.55)

ax.set_title(
    f"Knox County OD Graph — Top {TOP_N:,} Flows\n"
    "Node size = log(production)  |  Node color = net flow (red=generator, blue=attractor)\n"
    "Edge color = flow magnitude (yellow=low, red=high)",
    fontsize=11, fontweight="bold"
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(FIG_DIR / "figG1_od_graph_top_flows.png", dpi=160, bbox_inches="tight")
plt.close()
print("  Saved: figG1_od_graph_top_flows.png")

# =============================================================================
# Fig G2: OD Graph — ego network for top 5 producing zones
# =============================================================================
print("Fig G2: Ego networks for top 5 zones...")

top5 = od_stats.nlargest(5, "production")["zone_id"].tolist()

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, zid in zip(axes, top5):
    # All OD pairs from/to this zone
    zone_od = od_int[(od_int["origin"] == zid) | (od_int["destination"] == zid)]
    G_ego = nx.DiGraph()
    G_ego.add_nodes_from(G_od.nodes(data=True))
    for _, row in zone_od.iterrows():
        G_ego.add_edge(int(row["origin"]), int(row["destination"]), weight=float(row["flow"]))

    zones.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.2, zorder=1)

    ew = np.array([G_ego[u][v]["weight"] for u, v in G_ego.edges()])
    ew_norm = (ew - ew.min()) / (ew.max() - ew.min() + 1e-9)
    for (u, v), w in zip(G_ego.edges(), ew_norm):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        color = "red" if u == zid else "steelblue"
        ax.plot([x0, x1], [y0, y1], color=color,
                alpha=0.1 + w * 0.6, linewidth=0.3 + w * 1.5, zorder=2)

    # Highlight focal zone
    focal = zones[zones["zone_id"] == zid]
    focal.plot(ax=ax, color="gold", edgecolor="black", linewidth=1.0, zorder=3)

    prod = od_stats.loc[od_stats["zone_id"] == zid, "production"].values[0]
    ax.set_title(f"Zone {zid}\nprod={prod:,.0f}", fontsize=9, fontweight="bold")
    ax.set_axis_off()

fig.suptitle("Ego Networks: Top 5 Trip-Generating Zones\nRed = outgoing trips  |  Blue = incoming trips  |  Gold = focal zone",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figG2_ego_networks_top5.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figG2_ego_networks_top5.png")

# =============================================================================
# Fig G3a: Road network graph — Overture segments colored by road class
# =============================================================================
print("Fig G3a: Road network graph (Overture segments)...")
seg_path = OUT_DIR / "overture_cache" / "knox_segment.geojson"
if seg_path.exists():
    segs = gpd.read_file(seg_path).to_crs("EPSG:26917")
    segs["seg_length_m"] = segs.geometry.length

    fig, ax = plt.subplots(figsize=(14, 11))
    zones.plot(ax=ax, color="#f5f5f5", edgecolor="#dddddd", linewidth=0.3, zorder=1)
    if "class" in segs.columns:
        road_classes = sorted(segs["class"].fillna("unknown").unique())
        cmap_r = cm.get_cmap("tab20", len(road_classes))
        class_color = {c: cmap_r(i) for i, c in enumerate(road_classes)}
        for cls, grp in segs.groupby(segs["class"].fillna("unknown")):
            grp.plot(ax=ax, color=class_color[cls], linewidth=0.4,
                     alpha=0.7, zorder=2, label=cls)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8, loc="lower right",
                  title="Road class", title_fontsize=9,
                  framealpha=0.85, edgecolor="#aaaaaa")
    else:
        segs.plot(ax=ax, color="steelblue", linewidth=0.3, alpha=0.5, zorder=2)
    ax.set_title(
        f"Knox County Overture Road Network (2024)\n{len(segs):,} segments",
        fontsize=13, fontweight="bold"
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figG3a_road_network.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figG3a_road_network.png")

    # =========================================================================
    # Fig G3b: Street density choropleth
    # =========================================================================
    print("Fig G3b: Street density choropleth...")
    fig, ax = plt.subplots(figsize=(11, 9))
    clean = gpd.read_file(OUT_DIR / "zones_clean.gpkg")
    vmax_sd = clean["street_density_km_km2"].quantile(0.99)
    clean.assign(_v=clean["street_density_km_km2"].clip(upper=vmax_sd)).plot(
        column="_v", ax=ax, cmap="YlGn",
        legend=True, legend_kwds={"label": "Street density (km/km²)", "shrink": 0.6},
        edgecolor="white", linewidth=0.3, zorder=1,
        missing_kwds={"color": "lightgrey"}
    )
    ax.set_title(
        "Knox County Street Density by TAZ Zone\n(Overture Maps 2024 — km of road per km²)",
        fontsize=13, fontweight="bold"
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figG3b_street_density.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figG3b_street_density.png")
else:
    print("  Skipped figG3a/G3b — segment cache not found")

# =============================================================================
# Fig G4: Graph statistics summary
# =============================================================================
print("Fig G4: OD graph statistics...")

in_deg  = dict(G_od.in_degree(weight="weight"))
out_deg = dict(G_od.out_degree(weight="weight"))
degrees = pd.DataFrame({"in_strength": in_deg, "out_strength": out_deg})

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Degree distribution
ax = axes[0]
ax.hist(np.log1p(degrees["out_strength"]), bins=30, color="tomato", alpha=0.7, label="Out-strength")
ax.hist(np.log1p(degrees["in_strength"]),  bins=30, color="steelblue", alpha=0.7, label="In-strength")
ax.set_xlabel("log(weighted degree)", fontsize=10)
ax.set_ylabel("Number of zones", fontsize=10)
ax.set_title("OD Graph: Strength Distribution\n(weighted in/out degree)", fontsize=10, fontweight="bold")
ax.legend()

# Top 20 zones by production
ax = axes[1]
top20 = od_stats.nlargest(20, "production")
ax.barh(range(20), top20["production"].values, color="tomato", alpha=0.8)
ax.set_yticks(range(20))
ax.set_yticklabels([f"Zone {z}" for z in top20["zone_id"]], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Trip Productions", fontsize=10)
ax.set_title("Top 20 Trip-Generating Zones", fontsize=10, fontweight="bold")

# Production vs Attraction scatter (all zones)
ax = axes[2]
ax.scatter(od_stats["production"], od_stats["attraction"],
           s=12, alpha=0.5, color="purple", edgecolors="none")
lim = max(od_stats["production"].max(), od_stats["attraction"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="P = A line")
ax.set_xlabel("Production", fontsize=10)
ax.set_ylabel("Attraction", fontsize=10)
ax.set_title("Production vs Attraction\n(zones near diagonal = balanced)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)

fig.suptitle("Knox County OD Graph — Network Statistics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figG4_od_graph_stats.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figG4_od_graph_stats.png")

print(f"\nAll graph figures saved to: {FIG_DIR}")
print("Figures: figG1 (OD graph), figG2 (ego networks), figG3a (road network), figG3b (street density), figG4 (stats)")
