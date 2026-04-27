"""
Knox County City2Graph - Step 2b: Visualizations
=================================================
Choropleth maps + scatter plots from zones_clean.gpkg.
Saves all figures to outputs/figures/

Run with:
    .venv\Scripts\python.exe step2b_visualize.py
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

OUT_DIR  = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph\outputs")
FIG_DIR  = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load data
# =============================================================================
print("Loading zones_clean.gpkg...")
zones = gpd.read_file(OUT_DIR / "zones_clean.gpkg")

# Cap extreme outliers at 99th percentile for choropleth display
def cap99(col):
    p99 = zones[col].quantile(0.99)
    return zones[col].clip(upper=p99)

print(f"Zones loaded: {len(zones)}")
print(f"Columns: {zones.columns.tolist()}\n")

# =============================================================================
# Helper: single choropleth map
# =============================================================================
def choropleth(col, title, cmap="YlOrRd", cap=True, figsize=(10, 8), fname=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    vals = cap99(col) if cap else zones[col]
    zones.assign(_v=vals).plot(
        column="_v", ax=ax, cmap=cmap,
        legend=True,
        legend_kwds={"label": title, "shrink": 0.6},
        missing_kwds={"color": "lightgrey", "label": "No data"},
        edgecolor="white", linewidth=0.2
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    path = FIG_DIR / (fname or f"{col}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

# =============================================================================
# Fig 1: Trip Productions (total trips leaving each zone)
# =============================================================================
print("Generating choropleth maps...")
choropleth("production", "Trip Productions (2026)\nTotal trips leaving each TAZ zone",
           cmap="YlOrRd", fname="fig1_productions.png")

# =============================================================================
# Fig 2: Trip Attractions (total trips arriving)
# =============================================================================
choropleth("attraction", "Trip Attractions (2026)\nTotal trips arriving at each TAZ zone",
           cmap="Blues", fname="fig2_attractions.png")

# =============================================================================
# Fig 3: Net Flow (production - attraction)
# =============================================================================
# Diverging: negative = net attractor (employment), positive = net generator (residential)
fig, ax = plt.subplots(figsize=(10, 8))
vmax = zones["net_flow"].abs().quantile(0.97)
zones.plot(
    column="net_flow", ax=ax,
    cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    legend=True,
    legend_kwds={"label": "Net Flow (production - attraction)", "shrink": 0.6},
    edgecolor="white", linewidth=0.2
)
ax.set_title("Net Trip Flow by TAZ Zone (2026)\nBlue = net attractor (jobs/retail)   Red = net generator (residential)",
             fontsize=11, fontweight="bold")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_net_flow.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig3_net_flow.png")

# =============================================================================
# Fig 4: Street Density (km of road per km2)
# =============================================================================
choropleth("street_density_km_km2",
           "Street Density (km/km2)\nOverture Maps road network",
           cmap="Greens", fname="fig4_street_density.png")

# =============================================================================
# Fig 5: Building Density (buildings per km2)
# =============================================================================
choropleth("building_density_n_km2",
           "Building Density (buildings/km2)\nOverture Maps building footprints",
           cmap="Purples", fname="fig5_building_density.png")

# =============================================================================
# Fig 6: Building Coverage %
# =============================================================================
choropleth("building_coverage_pct",
           "Building Coverage (%)\nFootprint area / zone area",
           cmap="Oranges", fname="fig6_building_coverage.png")

# =============================================================================
# Fig 7: 2x3 Summary Map Panel
# =============================================================================
print("Generating summary panel...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
configs = [
    ("production",             "Trip Productions",        "YlOrRd"),
    ("attraction",             "Trip Attractions",        "Blues"),
    ("net_flow",               "Net Flow (prod - attr)",  "RdBu_r"),
    ("street_density_km_km2",  "Street Density (km/km2)", "Greens"),
    ("building_density_n_km2", "Building Density (n/km2)","Purples"),
    ("building_coverage_pct",  "Building Coverage (%)",   "Oranges"),
]
for ax, (col, title, cmap) in zip(axes.flat, configs):
    if col == "net_flow":
        vmax = zones[col].abs().quantile(0.97)
        zones.plot(column=col, ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax,
                   legend=False, edgecolor="white", linewidth=0.15)
    else:
        vals = cap99(col)
        zones.assign(_v=vals).plot(column="_v", ax=ax, cmap=cmap,
                                   legend=False, edgecolor="white", linewidth=0.15,
                                   missing_kwds={"color": "lightgrey"})
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_axis_off()
fig.suptitle("Knox County City2Graph - TAZ-level Summary (2026)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig7_summary_panel.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig7_summary_panel.png")

# =============================================================================
# Fig 8-10: Scatter plots — morphology vs trip generation
# =============================================================================
print("Generating scatter plots...")

def scatter(x_col, y_col, xlabel, ylabel, title, fname, log_x=True, log_y=True):
    df = zones[[x_col, y_col, "TOTPOP", "TOTAL_EMP"]].copy().dropna()
    df = df[(df[x_col] > 0) & (df[y_col] > 0)]

    x = np.log1p(df[x_col]) if log_x else df[x_col]
    y = np.log1p(df[y_col]) if log_y else df[y_col]

    # Color by total employment
    c = np.log1p(df["TOTAL_EMP"])
    corr = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=c, cmap="plasma", alpha=0.65, s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="log(Total Employment)")

    # Trend line
    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m * xline + b, "r--", linewidth=1.5, label=f"r = {corr:.3f}")

    ax.set_xlabel(("log " if log_x else "") + xlabel, fontsize=11)
    ax.set_ylabel(("log " if log_y else "") + ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}  (r = {corr:.3f})")

scatter("street_density_km_km2", "production",
        "Street Density (km/km2)", "Trip Productions",
        "Street Density vs Trip Productions\n(colored by Employment)",
        "fig8_street_density_vs_production.png")

scatter("building_density_n_km2", "production",
        "Building Density (n/km2)", "Trip Productions",
        "Building Density vs Trip Productions\n(colored by Employment)",
        "fig9_building_density_vs_production.png")

scatter("building_coverage_pct", "production",
        "Building Coverage (%)", "Trip Productions",
        "Building Coverage vs Trip Productions\n(colored by Employment)",
        "fig10_coverage_vs_production.png")

# =============================================================================
# Fig 11: Morphology vs Attraction (pull side)
# =============================================================================
scatter("street_density_km_km2", "attraction",
        "Street Density (km/km2)", "Trip Attractions",
        "Street Density vs Trip Attractions\n(colored by Employment)",
        "fig11_street_density_vs_attraction.png")

# =============================================================================
# Fig 12: Correlation matrix heatmap
# =============================================================================
print("Generating correlation heatmap...")
cols_of_interest = [
    "production", "attraction", "net_flow",
    "street_density_km_km2", "building_density_n_km2", "building_coverage_pct",
    "avg_seg_length_m", "n_segments", "n_buildings",
    "TOTPOP", "HH", "TOTAL_EMP", "area_km2"
]
# Only include columns that exist
cols_of_interest = [c for c in cols_of_interest if c in zones.columns]
corr_df = zones[cols_of_interest].apply(pd.to_numeric, errors="coerce").corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_xticks(range(len(cols_of_interest)))
ax.set_yticks(range(len(cols_of_interest)))
ax.set_xticklabels(cols_of_interest, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(cols_of_interest, fontsize=9)
for i in range(len(cols_of_interest)):
    for j in range(len(cols_of_interest)):
        v = corr_df.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(v) > 0.5 else "black")
ax.set_title("Correlation Matrix: Morphology + Socioeconomic + OD Variables",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig12_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig12_correlation_matrix.png")

# =============================================================================
# Print key correlation values for production
# =============================================================================
print("\n=== Key correlations with PRODUCTION ===")
prod_corr = corr_df["production"].drop("production").sort_values(key=abs, ascending=False)
print(prod_corr.to_string())

print("\n=== Key correlations with ATTRACTION ===")
attr_corr = corr_df["attraction"].drop("attraction").sort_values(key=abs, ascending=False)
print(attr_corr.to_string())

print(f"\nAll figures saved to: {FIG_DIR}")
print("Step 2b complete. Figures ready to inspect.")
