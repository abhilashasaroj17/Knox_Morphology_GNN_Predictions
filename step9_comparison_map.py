"""
Knox County — TPO Labels vs GNN Predictions on Overture Segments
================================================================
All layers are Overture road segments (65,524 total).

GROUP A — TPO-Labeled Segments (Overture segs matched to TPO volumes)
  Labels are purely from TPO volume/VC ground truth — independent of GNN.
  A1 · TPO Critical     (red)        — GT critical=1
  A2 · TPO Non-critical (light blue) — GT critical=0

GROUP B — Unlabeled Segments (no TPO match — pure GNN extension)
  GNN predicted on these 57k roads with no ground truth.
  B1 · GNN Critical, no TPO label            (orange)     — prob ≥ 0.50
  B2 · GNN Non-critical — High   0.30–0.50  (yellow-grey) — borderline
  B3 · GNN Non-critical — Medium 0.10–0.30  (light grey)
  B4 · GNN Non-critical — Low      < 0.10   (near-invisible, off by default)

Output: outputs/maps/map_gnn_completion.html
"""

import warnings; warnings.filterwarnings("ignore")
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MiniMap, Fullscreen
from pathlib import Path

ROOT     = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph")
CRIT_DIR = ROOT / "outputs" / "criticality"
OUT_HTML = ROOT / "outputs" / "maps"
OUT_HTML.mkdir(parents=True, exist_ok=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading Overture criticality segments...")
segs = gpd.read_file(CRIT_DIR / "critical_segments.gpkg").to_crs("EPSG:4326")
segs["pred_prob_critical"] = segs["pred_prob_critical"].fillna(0).astype(float)
segs["pred_critical"]      = segs["pred_critical"].fillna(0).astype(int)
print(f"  Total Overture segments: {len(segs):,}")

# ─── Segment masks ────────────────────────────────────────────────────────────
has_tpo  = segs["critical"].notna()
tpo_c    = has_tpo & (segs["critical"] == 1)    # TPO ground truth critical
tpo_nc   = has_tpo & (segs["critical"] == 0)    # TPO ground truth non-critical
gnn_only = ~has_tpo                              # no TPO label at all

p = segs["pred_prob_critical"]

# GNN-only segments split by predicted probability
gnn_crit  = gnn_only & (p >= 0.50)
gnn_nc_hi = gnn_only & (p >= 0.30) & (p < 0.50)
gnn_nc_md = gnn_only & (p >= 0.10) & (p < 0.30)
gnn_nc_lo = gnn_only & (p < 0.10)

# ─── Print counts ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("LAYER COUNTS")
print(f"{'='*60}")
print(f"GROUP A — TPO-labeled Overture segs ({int(has_tpo.sum()):,} total)")
print(f"  A1 · TPO Critical (GT=1):               {int(tpo_c.sum()):>6,}")
print(f"  A2 · TPO Non-critical (GT=0):            {int(tpo_nc.sum()):>6,}")
print(f"\nGROUP B — No TPO label ({int(gnn_only.sum()):,} total — GNN extension)")
print(f"  B1 · GNN Critical       (prob ≥ 0.50):  {int(gnn_crit.sum()):>6,}")
print(f"  B2 · GNN Non-crit High  (0.30–0.50):    {int(gnn_nc_hi.sum()):>6,}  ← borderline")
print(f"  B3 · GNN Non-crit Med   (0.10–0.30):    {int(gnn_nc_md.sum()):>6,}")
print(f"  B4 · GNN Non-crit Low   (< 0.10):       {int(gnn_nc_lo.sum()):>6,}")
print(f"{'='*60}")

# ─── Build map ────────────────────────────────────────────────────────────────
center = [segs.geometry.centroid.y.mean(), segs.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=12, tiles=None, prefer_canvas=True)

folium.TileLayer("CartoDB positron",    name="Light (default)", show=True ).add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Dark",            show=False).add_to(m)
folium.TileLayer("OpenStreetMap",       name="OSM",             show=False).add_to(m)

MiniMap(toggle_display=True).add_to(m)
Fullscreen().add_to(m)

fields  = ["class", "length_m", "volume", "pred_prob_critical", "critical"]
aliases = ["Road class", "Length (m)", "TPO volume", "GNN prob", "TPO label (GT)"]

def add_layer(mask, name, color, weight, opacity, show):
    """Add layer with all roads AND a separate non-service variant in LayerControl"""
    gdf = segs[mask].copy()
    
    # Layer 1: All roads (including service)
    label = f"{name}  [{len(gdf):,}]"
    fg = folium.FeatureGroup(name=label, show=show)
    if len(gdf) > 0:
        folium.GeoJson(
            gdf.__geo_interface__,
            style_function=lambda f, c=color, w=weight, o=opacity: {
                "color": c, "weight": w, "opacity": o,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=fields, aliases=aliases,
                localize=True, sticky=True, labels=True,
            ),
        ).add_to(fg)
    fg.add_to(m)
    print(f"  Added: {label}")
    
    # Layer 2: Non-service roads only
    mask_nonservice = mask & (segs["class"] != "service")
    gdf_ns = segs[mask_nonservice].copy()
    label_ns = f"{name} (no service)  [{len(gdf_ns):,}]"
    fg_ns = folium.FeatureGroup(name=label_ns, show=False)
    if len(gdf_ns) > 0:
        folium.GeoJson(
            gdf_ns.__geo_interface__,
            style_function=lambda f, c=color, w=weight, o=opacity: {
                "color": c, "weight": w, "opacity": o,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=fields, aliases=aliases,
                localize=True, sticky=True, labels=True,
            ),
        ).add_to(fg_ns)
    fg_ns.add_to(m)
    print(f"    ↳ Non-service variant: [{len(gdf_ns):,}]")

# GROUP A — TPO ground truth (draw non-critical first so critical renders on top)
print("\nAdding Group A (TPO ground truth)...")
add_layer(tpo_nc,    "A2 · TPO Non-critical (GT=0)",   "#A8D8EA", weight=2.0, opacity=0.65, show=True)
add_layer(tpo_c,     "A1 · TPO Critical (GT=1)",        "#C0392B", weight=4.0, opacity=0.95, show=True)

# GROUP B — GNN extension to unlabeled roads
print("\nAdding Group B (GNN extension — no TPO label)...")
add_layer(gnn_nc_lo, "B4 · GNN Non-critical Low  (<0.10)",       "#DEDEDE", weight=0.8, opacity=0.25, show=False)
add_layer(gnn_nc_md, "B3 · GNN Non-critical Med  (0.10–0.30)",   "#BBBBBB", weight=1.2, opacity=0.45, show=True)
add_layer(gnn_nc_hi, "B2 · GNN Non-critical High (0.30–0.50)",   "#E59866", weight=1.8, opacity=0.60, show=True)
add_layer(gnn_crit,  "B1 · GNN Critical, no TPO (prob ≥ 0.50)",  "#8E44AD", weight=3.0, opacity=0.90, show=True)

# ─── Legend ───────────────────────────────────────────────────────────────────
legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
            padding:16px 20px;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,0.3);
            font-family:Arial,sans-serif;font-size:12.5px;min-width:370px;line-height:2.0;">
  <b style="font-size:14px;">TPO Ground Truth vs GNN Extension — Overture Segments</b>
  <hr style="margin:8px 0;">
  <div style="font-size:11.5px;color:#666;margin-bottom:10px;font-style:italic;">
    💡 Use layer control (top-right) to toggle service roads for each layer
  </div>
  <div style="font-weight:bold;color:#333;">
    GROUP A — TPO-Labeled &nbsp;<span style="font-weight:normal;color:#888;font-size:11px;">({int(has_tpo.sum()):,} segs · labels from TPO volume/VC model)</span>
  </div>
  <div><span style="background:#C0392B;display:inline-block;width:28px;height:5px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    A1 · TPO Critical (GT=1) &nbsp;<b>{int(tpo_c.sum()):,}</b></div>
  <div><span style="background:#A8D8EA;display:inline-block;width:28px;height:4px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    A2 · TPO Non-critical (GT=0) &nbsp;<b>{int(tpo_nc.sum()):,}</b></div>
  <hr style="margin:8px 0;">
  <div style="font-weight:bold;color:#333;">
    GROUP B — No TPO Label &nbsp;<span style="font-weight:normal;color:#888;font-size:11px;">({int(gnn_only.sum()):,} segs · GNN prediction only)</span>
  </div>
  <div><span style="background:#8E44AD;display:inline-block;width:28px;height:5px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    B1 · GNN Critical, no TPO (≥0.50) &nbsp;<b>{int(gnn_crit.sum()):,}</b></div>
  <div><span style="background:#E59866;display:inline-block;width:28px;height:3px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    B2 · GNN Non-crit Borderline (0.30–0.50) &nbsp;<b>{int(gnn_nc_hi.sum()):,}</b></div>
  <div><span style="background:#BBBBBB;display:inline-block;width:28px;height:2px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    B3 · GNN Non-crit Medium (0.10–0.30) &nbsp;<b>{int(gnn_nc_md.sum()):,}</b></div>
  <div><span style="background:#DEDEDE;display:inline-block;width:28px;height:2px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    B4 · GNN Non-crit Low (&lt;0.10) &nbsp;<b>{int(gnn_nc_lo.sum()):,}</b>
    <span style="color:#aaa;font-size:11px;">(off by default)</span></div>
  <hr style="margin:6px 0;">
  <div style="font-size:11px;color:#666;">
    Total: <b>{len(segs):,}</b> &nbsp;|&nbsp;
    TPO-labeled: <b>{int(has_tpo.sum()):,}</b> &nbsp;|&nbsp;
    Unlabeled: <b>{int(gnn_only.sum()):,}</b>
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False).add_to(m)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = OUT_HTML / "map_gnn_completion.html"
m.save(str(out_path))
print(f"\nSaved: {out_path}")

