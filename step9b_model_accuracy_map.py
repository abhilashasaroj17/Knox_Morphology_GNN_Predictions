"""
Knox County — GNN Model Accuracy Map (Confusion Matrix Visualization)
======================================================================
Shows the 8,121 TPO-labeled Overture segments colored by model prediction outcome:
  - TP (True Positive):  TPO=Critical, GNN=Critical   — Model correctly identified
  - FN (False Negative): TPO=Critical, GNN=Non-crit   — Model missed critical road
  - FP (False Positive): TPO=Non-crit, GNN=Critical   — Model over-predicted
  - TN (True Negative):  TPO=Non-crit, GNN=Non-crit   — Model correctly identified

Each layer has a "(no service)" variant to toggle service roads.

Output: outputs/maps/map_model_accuracy.html
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

# ─── Filter to TPO-labeled segments only ─────────────────────────────────────
has_tpo = segs["critical"].notna()
labeled = segs[has_tpo].copy()
print(f"  TPO-labeled segments (for accuracy analysis): {len(labeled):,}")

# ─── Confusion matrix masks ──────────────────────────────────────────────────
tpo_crit = labeled["critical"] == 1
tpo_nc   = labeled["critical"] == 0
gnn_crit = labeled["pred_critical"] == 1
gnn_nc   = labeled["pred_critical"] == 0

tp_mask = tpo_crit & gnn_crit   # True Positive
fn_mask = tpo_crit & gnn_nc     # False Negative (missed critical)
fp_mask = tpo_nc & gnn_crit     # False Positive (over-predicted)
tn_mask = tpo_nc & gnn_nc       # True Negative

# ─── Print counts ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CONFUSION MATRIX — GNN Predictions on TPO-Labeled Segments")
print(f"{'='*60}")
print(f"  TP (TPO=1, GNN=1) — Correctly identified critical:  {int(tp_mask.sum()):>5,}")
print(f"  FN (TPO=1, GNN=0) — Missed critical roads:          {int(fn_mask.sum()):>5,}")
print(f"  FP (TPO=0, GNN=1) — Over-predicted as critical:     {int(fp_mask.sum()):>5,}")
print(f"  TN (TPO=0, GNN=0) — Correctly identified non-crit:  {int(tn_mask.sum()):>5,}")
print(f"{'='*60}")
tp_count = int(tp_mask.sum())
fn_count = int(fn_mask.sum())
fp_count = int(fp_mask.sum())
tn_count = int(tn_mask.sum())
precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
recall    = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy  = (tp_count + tn_count) / len(labeled)
print(f"  Accuracy:  {accuracy*100:.1f}%")
print(f"  Precision: {precision*100:.1f}%  (of GNN-critical, how many are truly critical)")
print(f"  Recall:    {recall*100:.1f}%  (of TPO-critical, how many did GNN catch)")
print(f"  F1 Score:  {f1:.3f}")
print(f"{'='*60}")

# ─── Build map ────────────────────────────────────────────────────────────────
center = [labeled.geometry.centroid.y.mean(), labeled.geometry.centroid.x.mean()]
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
    gdf = labeled[mask].copy()
    
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
    mask_nonservice = mask & (labeled["class"] != "service")
    gdf_ns = labeled[mask_nonservice].copy()
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

# Add layers in order: TN (background), then FP, FN, TP (most important on top)
print("\nAdding confusion matrix layers...")
add_layer(tn_mask, "TN · True Negative (TPO=0, GNN=0)",   "#B0BEC5", weight=1.5, opacity=0.35, show=True)
add_layer(fp_mask, "FP · False Positive (TPO=0, GNN=1)",  "#FFA726", weight=2.5, opacity=0.75, show=True)
add_layer(fn_mask, "FN · False Negative (TPO=1, GNN=0)",  "#FFEB3B", weight=3.0, opacity=0.85, show=True)
add_layer(tp_mask, "TP · True Positive (TPO=1, GNN=1)",   "#66BB6A", weight=4.0, opacity=0.95, show=True)

# ─── Legend ───────────────────────────────────────────────────────────────────
legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
            padding:16px 20px;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,0.3);
            font-family:Arial,sans-serif;font-size:12.5px;min-width:380px;line-height:2.0;">
  <b style="font-size:14px;">GNN Model Accuracy — Confusion Matrix</b>
  <hr style="margin:8px 0;">
  <div style="font-size:11.5px;color:#666;margin-bottom:10px;font-style:italic;">
    💡 Use layer control (top-right) to toggle service roads for each outcome
  </div>
  <div style="font-weight:bold;color:#333;margin-bottom:6px;">
    8,121 TPO-labeled segments (5-fold CV test predictions)
  </div>
  <div style="font-size:10.5px;color:#888;margin-bottom:8px;font-style:italic;">
    Each segment predicted when held-out from training (no train/test contamination)
  </div>
  <div><span style="background:#66BB6A;display:inline-block;width:28px;height:5px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    TP · True Positive (TPO=1, GNN=1) &nbsp;<b>{tp_count:,}</b></div>
  <div><span style="background:#FFEB3B;display:inline-block;width:28px;height:4px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    FN · False Negative (TPO=1, GNN=0) &nbsp;<b>{fn_count:,}</b> <span style="color:#d32f2f;font-size:11px;">← missed</span></div>
  <div><span style="background:#FFA726;display:inline-block;width:28px;height:4px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    FP · False Positive (TPO=0, GNN=1) &nbsp;<b>{fp_count:,}</b> <span style="color:#f57c00;font-size:11px;">← over-pred</span></div>
  <div><span style="background:#B0BEC5;display:inline-block;width:28px;height:2px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    TN · True Negative (TPO=0, GNN=0) &nbsp;<b>{tn_count:,}</b></div>
  <hr style="margin:8px 0;">
  <div style="font-size:11px;color:#444;line-height:1.7;">
    <b>Performance Metrics:</b><br>
    Accuracy:  <b>{accuracy*100:.1f}%</b> &nbsp;|&nbsp;
    Precision: <b>{precision*100:.1f}%</b><br>
    Recall:    <b>{recall*100:.1f}%</b> &nbsp;|&nbsp;
    F1 Score:  <b>{f1:.3f}</b>
  </div>
  <hr style="margin:8px 0;">
  <div style="font-size:10px;color:#888;">
    Model correctly identifies {tp_count:,} of {tp_count+fn_count:,} critical roads ({recall*100:.1f}% recall)
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False).add_to(m)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = OUT_HTML / "map_model_accuracy.html"
m.save(str(out_path))
print(f"\nSaved: {out_path}")
