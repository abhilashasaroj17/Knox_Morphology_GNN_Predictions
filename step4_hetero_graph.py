"""
Knox County City2Graph - Step 4: Heterogeneous Graph Construction
=================================================================
Builds a PyTorch Geometric HeteroData object with:

  Node type:
    'zone'  — 508 TAZ zones
              x        : normalised feature matrix (land use + morphology)
              y        : [production, attraction] (regression targets)
              zone_ids : original TAZID array (for mapping back)

  Edge types:
    ('zone', 'od_flow',   'zone') — directed OD edges, edge_attr = log(flow)
    ('zone', 'adjacent',  'zone') — undirected spatial contiguity edges
    ('zone', 'top_od',    'zone') — top-500 OD flows per zone (dense attention)

The object is saved as  outputs/knox_hetero_graph.pt

Also produces:
  figH1_node_feature_distributions.png
  figH2_graph_structure.png
  figH3_spatial_folds.png   (5-fold splits for GNN training)

Run with:
    $env:PYTHONUTF8="1"; .venv\\Scripts\\Activate.ps1; python step4_hetero_graph.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =============================================================================
# Paths
# =============================================================================
ROOT   = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph")
OUT    = ROOT / "outputs"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Section 1: Load all data
# =============================================================================
print("=" * 60)
print("  SECTION 1: Load data")
print("=" * 60)

zones  = gpd.read_file(OUT / "zones_clean.gpkg")
morph  = pd.read_csv(OUT / "morphology_zone_features.csv")
od_all = pd.read_csv(OUT / "od_long_2026_internal.csv")
od_stats = pd.read_csv(OUT / "od_zone_stats.csv")

# Merge everything into a per-zone dataframe
df = zones[["zone_id", "TOTPOP", "HH", "TOTAL_EMP", "geometry"]].copy()
df = df.merge(morph[["zone_id", "area_km2", "building_coverage_pct",
                      "avg_seg_length_m", "street_density_km_km2",
                      "building_density_n_km2", "avg_footprint_m2"]], on="zone_id", how="left")
df = df.merge(od_stats[["zone_id", "production", "attraction"]], on="zone_id", how="left")
df = df.fillna(0)
df = df.reset_index(drop=True)

print(f"  Zones: {len(df)}")
print(f"  OD pairs (internal): {len(od_all):,}")

# =============================================================================
# Section 2: Build zone→index mapping
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 2: Index mapping")
print("=" * 60)

zone_ids = df["zone_id"].values                      # original TAZID values
zone_to_idx = {zid: i for i, zid in enumerate(zone_ids)}
N = len(df)
print(f"  Node count: {N}")

# =============================================================================
# Section 3: Node feature matrix
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 3: Node features")
print("=" * 60)

# Feature columns — drop TOTPOP (VIF~10, collinear with HH)
FEAT_COLS = [
    # Land use
    "TOTAL_EMP", "HH", "area_km2",
    # Morphology
    "building_coverage_pct", "avg_seg_length_m",
    "street_density_km_km2", "building_density_n_km2", "avg_footprint_m2",
]
TARGET_COLS = ["production", "attraction"]

X_raw = df[FEAT_COLS].values.astype(np.float32)
y_raw = df[TARGET_COLS].values.astype(np.float32)

# Standardise features
scaler_x = StandardScaler()
X_norm = scaler_x.fit_transform(X_raw).astype(np.float32)

# Log-scale targets (less skew for regression loss)
y_log = np.log1p(y_raw).astype(np.float32)

print(f"  Feature matrix:  {X_norm.shape}  columns={FEAT_COLS}")
print(f"  Target matrix:   {y_log.shape}   columns={TARGET_COLS}")

# =============================================================================
# Section 4: OD flow edges  ('zone', 'od_flow', 'zone')
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 4: OD flow edges")
print("=" * 60)

# Map zone IDs to indices; drop pairs where either end is not in our node set
od_valid = od_all[
    od_all["origin"].isin(zone_to_idx) & od_all["destination"].isin(zone_to_idx)
].copy()
od_valid["src"] = od_valid["origin"].map(zone_to_idx).astype(int)
od_valid["dst"] = od_valid["destination"].map(zone_to_idx).astype(int)
od_valid = od_valid[od_valid["src"] != od_valid["dst"]]   # remove self-loops

src_od = torch.tensor(od_valid["src"].values, dtype=torch.long)
dst_od = torch.tensor(od_valid["dst"].values, dtype=torch.long)
edge_attr_od = torch.tensor(
    np.log1p(od_valid["flow"].values).astype(np.float32)
).unsqueeze(1)

print(f"  OD flow edges:  {src_od.shape[0]:,}")
print(f"  Edge attr shape: {edge_attr_od.shape}")

# Top-500 OD edges per origin zone (for dense attention layer in GNN)
top_od = od_all[od_all["origin"].isin(zone_to_idx) & od_all["destination"].isin(zone_to_idx)].copy()
top_od["src"] = top_od["origin"].map(zone_to_idx).astype(int)
top_od["dst"] = top_od["destination"].map(zone_to_idx).astype(int)
top_od = top_od[top_od["src"] != top_od["dst"]]
top_od = top_od.nlargest(500 * N // N, "flow")   # global top-500 per ratio
# Actually keep top 2000 globally by flow
top_od = top_od.nlargest(2000, "flow")
src_top = torch.tensor(top_od["src"].values, dtype=torch.long)
dst_top = torch.tensor(top_od["dst"].values, dtype=torch.long)
edge_attr_top = torch.tensor(
    np.log1p(top_od["flow"].values).astype(np.float32)
).unsqueeze(1)
print(f"  Top-OD edges:    {src_top.shape[0]:,}")

# =============================================================================
# Section 5: Spatial contiguity edges  ('zone', 'adjacent', 'zone')
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 5: Spatial contiguity edges")
print("=" * 60)

# Queen contiguity: zones that share any boundary point
zones_proj = df.copy()
zones_proj = gpd.GeoDataFrame(zones_proj, geometry="geometry")

adj_src, adj_dst = [], []
sindex = zones_proj.sindex
for i, geom in enumerate(zones_proj.geometry):
    candidates = list(sindex.intersection(geom.bounds))
    for j in candidates:
        if i != j and geom.touches(zones_proj.geometry.iloc[j]):
            adj_src.append(i)
            adj_dst.append(j)

src_adj = torch.tensor(adj_src, dtype=torch.long)
dst_adj = torch.tensor(adj_dst, dtype=torch.long)
print(f"  Contiguity edges: {src_adj.shape[0]:,}  (undirected pairs)")

# =============================================================================
# Section 6: Spatial 5-fold masks for GNN training
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 6: Spatial 5-fold masks")
print("=" * 60)

K = 5
cx = df.geometry.centroid.x.values
cy = df.geometry.centroid.y.values
km = KMeans(n_clusters=K, random_state=42, n_init=10)
fold_labels = km.fit_predict(np.column_stack([cx, cy]))

fold_masks = {}
for k in range(K):
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask[fold_labels == k]  = True
    train_mask[fold_labels != k] = True
    fold_masks[k] = {"train": train_mask, "test": test_mask}
    print(f"  Fold {k}: train={train_mask.sum().item()}  test={test_mask.sum().item()}")

# =============================================================================
# Section 7: Assemble HeteroData object
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 7: Assemble HeteroData")
print("=" * 60)

data = HeteroData()

# --- Nodes ---
data["zone"].x        = torch.tensor(X_norm, dtype=torch.float)
data["zone"].y        = torch.tensor(y_log,  dtype=torch.float)
data["zone"].y_raw    = torch.tensor(y_raw,  dtype=torch.float)  # original scale
data["zone"].zone_ids = torch.tensor(zone_ids, dtype=torch.long)
data["zone"].fold     = torch.tensor(fold_labels, dtype=torch.long)

# Store scaler stats for inverse transform
data["zone"].feat_mean = torch.tensor(scaler_x.mean_.astype(np.float32))
data["zone"].feat_std  = torch.tensor(scaler_x.scale_.astype(np.float32))

# --- Edges ---
data["zone", "od_flow",  "zone"].edge_index = torch.stack([src_od, dst_od], dim=0)
data["zone", "od_flow",  "zone"].edge_attr  = edge_attr_od

data["zone", "top_od",   "zone"].edge_index = torch.stack([src_top, dst_top], dim=0)
data["zone", "top_od",   "zone"].edge_attr  = edge_attr_top

data["zone", "adjacent", "zone"].edge_index = torch.stack([src_adj, dst_adj], dim=0)

# Save fold masks alongside data
data["zone"].fold_masks = fold_masks   # dict of {k: {train, test}}

print(f"\n  HeteroData summary:")
print(f"    zone nodes:         {data['zone'].x.shape}")
print(f"    zone targets:       {data['zone'].y.shape}")
print(f"    od_flow edges:      {data['zone','od_flow','zone'].edge_index.shape[1]:,}")
print(f"    top_od edges:       {data['zone','top_od','zone'].edge_index.shape[1]:,}")
print(f"    adjacent edges:     {data['zone','adjacent','zone'].edge_index.shape[1]:,}")

# =============================================================================
# Section 8: Save
# =============================================================================
save_path = OUT / "knox_hetero_graph.pt"
torch.save(data, save_path)
print(f"\n  Saved: {save_path}")

# Also save feature column names and zone_id array as metadata
meta = {
    "feat_cols":    FEAT_COLS,
    "target_cols":  TARGET_COLS,
    "n_zones":      N,
    "n_folds":      K,
}
import json
(OUT / "graph_metadata.json").write_text(json.dumps(meta, indent=2))
print(f"  Saved: graph_metadata.json")

# =============================================================================
# Section 9: Figures
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 9: Figures")
print("=" * 60)

# --- Fig H1: Node feature distributions ---
print("  Fig H1: Feature distributions...")
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(FEAT_COLS):
    ax = axes[i]
    vals = df[col].values
    ax.hist(vals[vals > 0], bins=40, color="#2C3E50", alpha=0.75, edgecolor="white")
    ax.set_title(col, fontsize=9, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    # Annotate
    ax.axvline(np.median(vals), color="red", linewidth=1.2, linestyle="--",
               label=f"median={np.median(vals):.1f}")
    ax.legend(fontsize=7)
fig.suptitle("Knox County — Node Feature Distributions (raw, zones > 0)\n"
             "Red dashed = median", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figH1_node_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figH1_node_feature_distributions.png")

# --- Fig H2: Graph structure summary ---
print("  Fig H2: Graph structure...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: out-degree distribution (OD graph)
ax = axes[0]
out_deg = pd.Series(od_valid["src"].values).value_counts().reindex(range(N), fill_value=0)
ax.hist(out_deg.values, bins=40, color="#E74C3C", alpha=0.75, edgecolor="white")
ax.set_xlabel("Out-degree (number of destinations)", fontsize=10)
ax.set_ylabel("Number of zones", fontsize=10)
ax.set_title("OD Graph — Out-degree Distribution", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# Middle: edge weight distribution (log flow)
ax = axes[1]
ax.hist(edge_attr_od.numpy().flatten(), bins=50, color="#3498DB", alpha=0.75, edgecolor="white")
ax.set_xlabel("log(OD flow + 1)", fontsize=10)
ax.set_ylabel("Edge count", fontsize=10)
ax.set_title("OD Edge Weight Distribution\n(log scale)", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# Right: adjacency degree
ax = axes[2]
adj_deg = pd.Series(adj_src).value_counts().reindex(range(N), fill_value=0)
ax.hist(adj_deg.values, bins=20, color="#2ECC71", alpha=0.75, edgecolor="white")
ax.set_xlabel("Number of contiguous neighbors", fontsize=10)
ax.set_ylabel("Number of zones", fontsize=10)
ax.set_title("Spatial Adjacency Degree", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

fig.suptitle("Knox County HeteroData — Graph Structure Summary\n"
             f"Nodes: {N}  |  OD edges: {src_od.shape[0]:,}  |  Adjacency edges: {src_adj.shape[0]:,}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figH2_graph_structure.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figH2_graph_structure.png")

# --- Fig H3: Spatial folds map ---
print("  Fig H3: Spatial folds...")
fold_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
fig, ax = plt.subplots(figsize=(11, 9))
for k in range(K):
    mask = fold_labels == k
    subset = gpd.GeoDataFrame(df[mask], geometry="geometry")
    subset.plot(ax=ax, color=fold_colors[k], edgecolor="white", linewidth=0.3,
                alpha=0.75, zorder=2)
ax.set_title("Spatial 5-Fold Cross-Validation Splits\n"
             "Each color = one test fold (trained on all other colors)",
             fontsize=12, fontweight="bold")
ax.set_axis_off()
legend_patches = [mpatches.Patch(color=fold_colors[k],
                                  label=f"Fold {k}  (n={int((fold_labels==k).sum())})")
                  for k in range(K)]
ax.legend(handles=legend_patches, fontsize=10, loc="lower right",
          title="Folds", title_fontsize=11, framealpha=0.85)
plt.tight_layout()
plt.savefig(FIG_DIR / "figH3_spatial_folds.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figH3_spatial_folds.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 4 COMPLETE")
print("=" * 60)
print(f"""
  HeteroData saved:  outputs/knox_hetero_graph.pt
  Metadata saved:    outputs/graph_metadata.json

  Graph contents:
    Nodes  — zone:     {N} × {len(FEAT_COLS)} features
    Edges  — od_flow:  {src_od.shape[0]:,} directed (full OD)
    Edges  — top_od:   {src_top.shape[0]:,} directed (top 2000 flows)
    Edges  — adjacent: {src_adj.shape[0]:,} undirected (contiguity)

  Target: log(production), log(attraction) per zone
  Folds:  {K} spatial folds (KMeans on centroids)

  Feature columns (normalised):
    {FEAT_COLS}

Next: step5_gnn_train.py
""")
