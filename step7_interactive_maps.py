"""
Knox County City2Graph - Step 7: Interactive HTML Maps
=======================================================
Produces two standalone HTML maps using Folium + GeoJSON:

  Map 1 — Critical roads only
    Roads colored by predicted criticality probability (0–1)
    Toggle layers: Critical (predicted) | Ground truth labeled | All roads

  Map 2 — Full network heatmap
    All 65,524 driveable segments colored on a continuous scale
    by pred_prob_critical, from grey (safe) → red (critical)

Run with:
    $env:PYTHONUTF8="1"; .venv\\Scripts\\Activate.ps1; python step7_interactive_maps.py
"""

import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import Search, MiniMap, Fullscreen, MarkerCluster
from branca.colormap import LinearColormap
import json
from pathlib import Path

ROOT     = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph")
CRIT_DIR = ROOT / "outputs" / "criticality"
OUT_HTML = ROOT / "outputs" / "maps"
OUT_HTML.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:4326"   # Folium needs WGS84

# =============================================================================
# Load data
# =============================================================================
print("Loading criticality results...")
segs = gpd.read_file(CRIT_DIR / "critical_segments.gpkg").to_crs(TARGET_CRS)

# Fill any NaN probs
segs["pred_prob_critical"]  = segs["pred_prob_critical"].fillna(0).astype(float)
segs["pred_critical"]       = segs["pred_critical"].fillna(0).astype(int)
segs["has_ground_truth"]    = segs["critical"].notna()
segs["critical_gt"]         = segs["critical"].fillna(-1).astype(int)

# Geometry centroid for map center
cx = segs.geometry.centroid.x.mean()
cy = segs.geometry.centroid.y.mean()
center = [cy, cx]

print(f"  Segments loaded: {len(segs):,}")
print(f"  Map center: {center[0]:.4f}, {center[1]:.4f}")

# =============================================================================
# Helper: segment color by probability
# =============================================================================
def prob_to_hex(p, alpha_hex=""):
    """Map probability 0–1 → hex color grey→yellow→orange→red."""
    p = float(np.clip(p, 0, 1))
    if p < 0.3:
        r, g, b = 150, 150, 150          # grey
    elif p < 0.5:
        t = (p - 0.3) / 0.2
        r = int(150 + t * (255 - 150))
        g = int(150 + t * (200 - 150))
        b = int(150 * (1 - t))
    elif p < 0.75:
        t = (p - 0.5) / 0.25
        r = 255
        g = int(200 - t * 150)
        b = 0
    else:
        r, g, b = 220, 20, 20            # red
    return f"#{r:02x}{g:02x}{b:02x}"

def line_weight(p):
    return 1.5 + p * 4.0   # thin for low prob, thick for high

# =============================================================================
# Map 1 — Critical roads only (with layer control)
# =============================================================================
print("\nBuilding Map 1: Critical roads (layered)...")

m1 = folium.Map(
    location=center,
    zoom_start=11,
    tiles=None,
)

# Base tile layers
folium.TileLayer("CartoDB positron", name="Light (CartoDB)", control=True).add_to(m1)
folium.TileLayer("CartoDB dark_matter", name="Dark (CartoDB)", control=True).add_to(m1)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m1)

Fullscreen().add_to(m1)
MiniMap(toggle_display=True).add_to(m1)

# --- Layer A: All driveable roads (thin grey background) ---
print("  Adding background roads...")
bg_group = folium.FeatureGroup(name="All driveable roads (background)", show=True)
for _, row in segs.iterrows():
    p = row["pred_prob_critical"]
    if p < 0.2:   # skip very low-prob to keep file small
        coords = list(row.geometry.coords)
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in coords],
            color="#cccccc", weight=0.8, opacity=0.4,
        ).add_to(bg_group)
bg_group.add_to(m1)

# --- Layer B1: Predicted critical non-service (prob > 0.2) ---
print("  Adding predicted critical roads (non-service)...")
crit_group = folium.FeatureGroup(name="Predicted critical (non-service)", show=True)
crit_svc_group = folium.FeatureGroup(name="Predicted critical (service roads)", show=True)

crit_segs = segs[segs["pred_prob_critical"] >= 0.2].copy()
for _, row in crit_segs.iterrows():
    p   = row["pred_prob_critical"]
    is_svc = (str(row.get("class", "")).lower() == "service")
    col = "#9B59B6" if is_svc else prob_to_hex(p)
    wt  = 2.0 if is_svc else line_weight(p)
    coords = list(row.geometry.coords)
    tooltip = (
        f"<b>Class:</b> {row.get('class','?')}<br>"
        f"<b>Criticality prob:</b> {p:.3f}<br>"
        f"<b>Predicted critical:</b> {'Yes' if row['pred_critical']==1 else 'No'}<br>"
        f"<b>Length:</b> {row.get('length_m', 0):.0f} m<br>"
        f"<b>Betweenness:</b> {row.get('betweenness', 0):.5f}<br>"
        f"<b>Volume (TPO):</b> {row.get('volume', 'n/a')}<br>"
        f"<b>Graph degree:</b> {row.get('graph_degree', '?')}"
    )
    target_group = crit_svc_group if is_svc else crit_group
    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in coords],
        color=col, weight=wt, opacity=0.85,
        tooltip=folium.Tooltip(tooltip, sticky=True),
    ).add_to(target_group)
crit_group.add_to(m1)
crit_svc_group.add_to(m1)

# --- Layer C: Ground truth critical (TPO-labeled only) ---
print("  Adding ground truth critical roads...")
gt_group = folium.FeatureGroup(name="Ground truth critical (TPO-labeled)", show=False)
gt_crit = segs[(segs["has_ground_truth"]) & (segs["critical_gt"] == 1)]
for _, row in gt_crit.iterrows():
    coords = list(row.geometry.coords)
    tooltip = (
        f"<b>GROUND TRUTH CRITICAL</b><br>"
        f"<b>Class:</b> {row.get('class','?')}<br>"
        f"<b>Criticality score:</b> {row.get('criticality_score', 0):.4f}<br>"
        f"<b>Volume (TPO):</b> {row.get('volume', 'n/a')}<br>"
        f"<b>Pred prob:</b> {row.get('pred_prob_critical', 0):.3f}"
    )
    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in coords],
        color="#0000ff", weight=2.5, opacity=0.9,
        tooltip=folium.Tooltip(tooltip, sticky=True),
    ).add_to(gt_group)
gt_group.add_to(m1)

# Colorbar legend
colormap1 = LinearColormap(
    colors=["#969696", "#ffcc00", "#ff6600", "#dc1414"],
    vmin=0, vmax=1,
    caption="Predicted criticality probability"
)
colormap1.add_to(m1)

folium.LayerControl(position="topright", collapsed=False).add_to(m1)

# Title overlay
title_html = """
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
     z-index: 1000; background: white; padding: 8px 16px; border-radius: 6px;
     box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial; font-size: 14px;
     font-weight: bold; color: #333;">
  Knox County — Road Criticality GNN (GAT)
  <span style="font-weight:normal; font-size:11px; color:#666;">
    &nbsp;|&nbsp; Red = predicted critical &nbsp;|&nbsp; Blue = ground truth critical
  </span>
</div>
"""
m1.get_root().html.add_child(folium.Element(title_html))

out1 = OUT_HTML / "map1_critical_roads.html"
m1.save(str(out1))
print(f"  Saved: {out1}")

# =============================================================================
# Map 2 — Full network heatmap (all 65k segments)
# =============================================================================
print("\nBuilding Map 2: Full network heatmap...")

# To keep file size reasonable, simplify geometry slightly and
# split into probability bands with shared style
m2 = folium.Map(
    location=center,
    zoom_start=11,
    tiles=None,
)
folium.TileLayer("CartoDB positron", name="Light (CartoDB)", control=True).add_to(m2)
folium.TileLayer("CartoDB dark_matter", name="Dark (CartoDB)", control=True).add_to(m2)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m2)

Fullscreen().add_to(m2)
MiniMap(toggle_display=True).add_to(m2)

# Define probability bands → group into layers to keep file manageable
bands = [
    ("Very low (0–0.2)",  0.00, 0.20, "#bbbbbb", 0.6, 0.7, True),
    ("Low (0.2–0.4)",     0.20, 0.40, "#ffcc44", 1.0, 0.8, True),
    ("Medium (0.4–0.6)",  0.40, 0.60, "#ff8800", 1.5, 0.85, True),
    ("High (0.6–0.8)",    0.60, 0.80, "#ff3300", 2.5, 0.9, True),
    ("Critical (0.8–1)",  0.80, 1.01, "#cc0000", 3.5, 1.0, True),
]

for label, lo, hi, color, weight, opacity, show in bands:
    band_segs = segs[(segs["pred_prob_critical"] >= lo) & (segs["pred_prob_critical"] < hi)]
    grp = folium.FeatureGroup(name=f"<span style='color:{color}'>{label}</span> ({len(band_segs):,} segments)", show=show)
    for _, row in band_segs.iterrows():
        coords = list(row.geometry.coords)
        p = row["pred_prob_critical"]
        tooltip = (
            f"<b>Class:</b> {row.get('class','?')}<br>"
            f"<b>Criticality prob:</b> {p:.3f}<br>"
            f"<b>Length:</b> {row.get('length_m', 0):.0f} m<br>"
            f"<b>Betweenness:</b> {row.get('betweenness', 0):.5f}<br>"
            f"<b>Volume (TPO):</b> {row.get('volume', 0) if row.get('volume', 0) > 0 else 'unlabeled'}"
        )
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in coords],
            color=color, weight=weight, opacity=opacity,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(grp)
    grp.add_to(m2)
    print(f"  Band '{label}': {len(band_segs):,} segments")

# Colorbar
colormap2 = LinearColormap(
    colors=["#bbbbbb", "#ffcc44", "#ff8800", "#ff3300", "#cc0000"],
    vmin=0, vmax=1,
    caption="Predicted criticality probability (all 65,524 segments)"
)
colormap2.add_to(m2)

folium.LayerControl(position="topright", collapsed=False).add_to(m2)

title_html2 = """
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
     z-index: 1000; background: white; padding: 8px 16px; border-radius: 6px;
     box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial; font-size: 14px;
     font-weight: bold; color: #333;">
  Knox County — Full Road Network Criticality Heatmap
  <span style="font-weight:normal; font-size:11px; color:#666;">
    &nbsp;|&nbsp; All 65,524 Overture segments &nbsp;|&nbsp; Grey → Red = criticality probability
  </span>
</div>
"""
m2.get_root().html.add_child(folium.Element(title_html2))

out2 = OUT_HTML / "map2_full_heatmap.html"
m2.save(str(out2))
print(f"  Saved: {out2}")

# =============================================================================
print(f"""
============================================================
  STEP 7 COMPLETE
============================================================
  Map 1 (critical roads, layered):  {out1}
  Map 2 (full network heatmap):     {out2}

  Open either file in a browser — no server needed.
  Both maps support:
    - Click/hover tooltips per segment
    - Layer toggles (top right)
    - Mini-map + fullscreen
    - Multiple base tile styles
""")
