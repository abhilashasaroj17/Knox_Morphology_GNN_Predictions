"""
Knox County City2Graph - Step 6: Road Criticality GNN
======================================================
Predicts which road segments in Knox County are "critical infrastructure"
using a GNN trained on TPO-assigned volumes + graph topology.

Pipeline:
  1. Build driveable road graph from Overture segments
     Nodes = segments (line graph), Edges = shared connector adjacency
  2. Match TPO assignment links → Overture segments (nearest spatial join)
  3. Compute approximate betweenness centrality on road graph
  4. Criticality score = 0.5 * norm(volume) + 0.5 * norm(betweenness)
     Label = top-20% by score → critical (1)
  5. Node features (all 65k segments — Overture-native + TAZ morphology):
                    road class, length, speed limit, connector count,
                    surface flag, graph degree, betweenness centrality,
                    TAZ employment, households, building coverage,
                    street density, building density, avg footprint
     NOTE: TPO-derived fields (volume, V/C, lanes) are NOT node features —
     they are only used to define the training label (criticality score).
     This is intentional: the 61k unlabeled segments have no volume data,
     so the model must generalize from topology + land use alone.
  6. Train GAT (Graph Attention Network) — spatial 5-fold CV on labeled nodes
  7. Predict criticality on ALL 73,502 Overture segments
  8. Export criticality map (GeoPackage + figures)

Outputs (outputs/criticality/):
  - road_graph.pt               — PyG Data object (full road network)
  - criticality_scores.csv      — all segments with scores + predicted labels
  - critical_segments.gpkg      — GeoPackage of predicted critical roads
  - figC1_criticality_score_distribution.png
  - figC2_criticality_map.png
  - figC3_gnn_roc_curve.png
  - figC4_feature_importance.png
  - cv_criticality_results.csv

Run with:
    $env:PYTHONUTF8="1"; .venv\\Scripts\\Activate.ps1; python step6_road_criticality.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json, copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, roc_curve, confusion_matrix)
from sklearn.cluster import KMeans

# =============================================================================
# Config
# =============================================================================
ROOT     = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph")
OUT      = ROOT / "outputs"
CRIT_DIR = OUT / "criticality"
FIG_DIR  = OUT / "figures"
CRIT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS   = "EPSG:26917"
MATCH_DIST_M = 150         # max distance to match TPO link → Overture segment (line-to-line)
CRIT_PCTILE  = 80          # top-X% by combined score = critical
ALPHA        = 0.5         # weight: volume vs betweenness
BETW_K       = 300         # betweenness approximation samples
EPOCHS       = 400
LR           = 5e-4
HIDDEN       = 64
HEADS        = 4
DROPOUT      = 0.4
PATIENCE     = 40
K_FOLDS      = 5

# Driveable road classes only (exclude footways, rail, steps, cycleways)
DRIVEABLE = {"motorway","trunk","primary","secondary","tertiary",
             "residential","service","living_street","unclassified","unknown"}

# Road class priority (higher = more important a priori)
CLASS_ORDER = ["motorway","trunk","primary","secondary","tertiary",
               "unclassified","unknown","living_street","residential","service"]

# =============================================================================
# Section 1: Load Overture segments — driveable only
# =============================================================================
print("=" * 60)
print("  SECTION 1: Load Overture segments")
print("=" * 60)

segs_raw = gpd.read_file(OUT / "overture_cache" / "knox_segment.geojson")
segs = segs_raw[segs_raw["class"].isin(DRIVEABLE)].copy()
segs = segs.to_crs(TARGET_CRS)
segs = segs.reset_index(drop=True)
segs["seg_idx"] = segs.index                         # stable integer index
segs["length_m"] = segs.geometry.length

print(f"  Driveable segments: {len(segs):,} / {len(segs_raw):,} total")
print(f"  Class distribution:\n{segs['class'].value_counts().to_string()}")

# =============================================================================
# Section 2: Parse connector degree per segment
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 2: Connector degree")
print("=" * 60)

def parse_connector_count(val):
    """Count connectors listed in the connectors JSON column."""
    try:
        if val is None:
            return 1
        if isinstance(val, (list, np.ndarray)):
            return len(val)
        if isinstance(val, str):
            if val == "":
                return 1
            parsed = json.loads(val)
            return len(parsed) if isinstance(parsed, list) else 1
        if isinstance(val, float) and np.isnan(val):
            return 1
    except Exception:
        pass
    return 1

segs["connector_count"] = segs["connectors"].apply(parse_connector_count)
print(f"  Avg connectors per segment: {segs['connector_count'].mean():.2f}")

def parse_speed_limit(val):
    """Extract numeric speed limit (km/h) from speed_limits JSON."""
    try:
        if val is None:
            return np.nan
        if isinstance(val, float) and np.isnan(val):
            return np.nan
        if isinstance(val, (list, np.ndarray)):
            parsed = val
        elif isinstance(val, str):
            if val == "":
                return np.nan
            parsed = json.loads(val)
        else:
            return np.nan
        if isinstance(parsed, list) and len(parsed) > 0:
            first = parsed[0]
            if isinstance(first, dict):
                v = first.get("max_speed", {})
                if isinstance(v, dict):
                    return float(v.get("value", np.nan))
    except Exception:
        pass
    return np.nan

segs["speed_kph"] = segs["speed_limits"].apply(parse_speed_limit)
# Fill missing speed by class median
speed_medians = segs.groupby("class")["speed_kph"].median()
segs["speed_kph"] = segs.apply(
    lambda r: speed_medians.get(r["class"], 50) if np.isnan(r["speed_kph"]) else r["speed_kph"],
    axis=1
)

# Surface encoding
def is_surface_present(val):
    if val is None:
        return 0
    if isinstance(val, float) and np.isnan(val):
        return 0
    if isinstance(val, str) and val.strip() in ("", "[]", "null"):
        return 0
    if isinstance(val, (list, np.ndarray)) and len(val) == 0:
        return 0
    return 1

segs["has_surface"] = segs["road_surface"].apply(is_surface_present)

# Class encoding
le = LabelEncoder()
segs["class_enc"] = le.fit_transform(segs["class"].fillna("unknown"))
print(f"  Road classes encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- road_flags binary features: is_bridge, is_link, is_tunnel ---
def parse_road_flag(val, flag):
    """Return 1 if the given flag string appears in the road_flags JSON."""
    try:
        if val is None:
            return 0
        s = val if isinstance(val, str) else json.dumps(val)
        return 1 if flag in s else 0
    except Exception:
        return 0

segs["is_bridge"] = segs["road_flags"].apply(lambda v: parse_road_flag(v, "is_bridge"))
segs["is_link"]   = segs["road_flags"].apply(lambda v: parse_road_flag(v, "is_link"))
segs["is_tunnel"] = segs["road_flags"].apply(lambda v: parse_road_flag(v, "is_tunnel"))

# --- access_restrictions: is_private ---
segs["is_private"] = segs["access_restrictions"].apply(
    lambda v: parse_road_flag(v, "as_private")
)

# --- sinuosity: actual length / straight-line distance (1.0 = perfectly straight) ---
from shapely.geometry import LineString
def sinuosity(geom):
    try:
        coords = list(geom.coords)
        if len(coords) < 2:
            return 1.0
        straight = geom.coords[0][0:2], geom.coords[-1][0:2]
        dx = straight[1][0] - straight[0][0]
        dy = straight[1][1] - straight[0][1]
        eucl = (dx**2 + dy**2) ** 0.5
        return geom.length / eucl if eucl > 0 else 1.0
    except Exception:
        return 1.0

segs["sinuosity"] = segs.geometry.apply(sinuosity)

print(f"  Bridges: {segs['is_bridge'].sum():,}  Links/ramps: {segs['is_link'].sum():,}  "
      f"Tunnels: {segs['is_tunnel'].sum():,}  Private: {segs['is_private'].sum():,}")
print(f"  Sinuosity range: {segs['sinuosity'].min():.2f} – {segs['sinuosity'].max():.2f}  "
      f"median: {segs['sinuosity'].median():.3f}")

# =============================================================================
# Section 3: Build road adjacency graph (line graph)
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 3: Build road adjacency (line graph)")
print("=" * 60)

# Two segments are adjacent if they share a connector endpoint
# Use a spatial join: endpoint buffers intersect
# For speed: extract start/end points, buffer 5m, sjoin to other endpoints

# Extract endpoints as points
from shapely.geometry import Point

endpoints = []
for idx, row in segs.iterrows():
    geom = row.geometry
    coords = list(geom.coords)
    endpoints.append({"seg_idx": idx, "end": "start", "geometry": Point(coords[0])})
    endpoints.append({"seg_idx": idx, "end": "end",   "geometry": Point(coords[-1])})

ep_gdf = gpd.GeoDataFrame(endpoints, crs=TARGET_CRS)
ep_buf = ep_gdf.copy()
ep_buf["geometry"] = ep_buf.geometry.buffer(8)   # 8m tolerance

# Self-join to find segments sharing an endpoint region
joined = gpd.sjoin(
    ep_gdf[["seg_idx","geometry"]].rename(columns={"seg_idx":"seg_a"}),
    ep_buf[["seg_idx","geometry"]].rename(columns={"seg_idx":"seg_b"}),
    how="inner", predicate="within"
)
joined = joined[joined["seg_a"] != joined["seg_b"]]
adj_pairs = joined[["seg_a","seg_b"]].drop_duplicates().reset_index(drop=True)

src_adj = torch.tensor(adj_pairs["seg_a"].values, dtype=torch.long)
dst_adj = torch.tensor(adj_pairs["seg_b"].values, dtype=torch.long)
edge_index = torch.stack([src_adj, dst_adj], dim=0)

print(f"  Road graph nodes (segments): {len(segs):,}")
print(f"  Road graph edges (adjacency): {edge_index.shape[1]:,}")

# Degree per segment node
deg = pd.Series(adj_pairs["seg_a"].values).value_counts().reindex(segs.index, fill_value=0)
segs["graph_degree"] = deg.values

# =============================================================================
# Section 4: Betweenness centrality (approximate)
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 4: Betweenness centrality (k={})".format(BETW_K))
print("=" * 60)

# Build networkx graph for betweenness
G_road = nx.Graph()
G_road.add_nodes_from(segs.index.tolist())
G_road.add_edges_from(zip(adj_pairs["seg_a"].tolist(), adj_pairs["seg_b"].tolist()))

print(f"  Computing approximate betweenness (k={BETW_K} samples)...")
betw = nx.betweenness_centrality(G_road, k=BETW_K, normalized=True, seed=42)
segs["betweenness"] = segs.index.map(betw).fillna(0)
print(f"  Betweenness range: {segs['betweenness'].min():.6f} – {segs['betweenness'].max():.6f}")

# =============================================================================
# Section 5: Match TPO assignment → Overture segments
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 5: Match TPO → Overture segments")
print("=" * 60)

assign = gpd.read_file(OUT / "assignment_with_volumes.gpkg").to_crs(TARGET_CRS)
assign["volume"] = pd.to_numeric(assign["Tot_Flow"], errors="coerce").fillna(0)
voc_col = "Max_VOC" if "Max_VOC" in assign.columns else ("AB_VOC" if "AB_VOC" in assign.columns else None)
assign["voc"]    = pd.to_numeric(assign[voc_col], errors="coerce").fillna(0) if voc_col else 0
spd_col = "AB_Speed" if "AB_Speed" in assign.columns else None
assign["speed"]  = pd.to_numeric(assign[spd_col], errors="coerce").fillna(0) if spd_col else 0

# ── Two-pass matching ────────────────────────────────────────────────────────
# Pass 1: midpoint-to-midpoint (fast, tight 50m — catches easily aligned links)
# Pass 2: full line-to-line geometry distance for remaining unmatched TPO links
#         (handles cases where long segments have midpoints far apart but overlap)

segs_lines = segs[["seg_idx","geometry"]].copy()
segs_mid   = segs[["seg_idx","geometry"]].copy()
segs_mid["geometry"] = segs_mid.geometry.interpolate(0.5, normalized=True)

assign_mid = assign[["volume","voc","speed","geometry"]].copy()
assign_mid["geometry"] = assign_mid.geometry.interpolate(0.5, normalized=True)
assign_mid["_tpo_idx"] = np.arange(len(assign_mid))
assign_lines = assign[["volume","voc","speed","geometry"]].copy()
assign_lines["_tpo_idx"] = np.arange(len(assign_lines))

# Pass 1: midpoint match at 50m
m1 = gpd.sjoin_nearest(
    assign_mid,
    segs_mid,
    how="left",
    max_distance=50,
    distance_col="match_dist"
)
matched_p1    = m1.dropna(subset=["seg_idx"]).copy()
unmatched_idx = m1[m1["seg_idx"].isna()]["_tpo_idx"].values
print(f"  Pass 1 (midpoint ≤50m): {len(matched_p1):,} matched | {len(unmatched_idx):,} unmatched")

# Pass 2: full line-to-line match for remaining, up to MATCH_DIST_M
unmatched_lines = assign_lines[assign_lines["_tpo_idx"].isin(unmatched_idx)].copy()
m2 = gpd.sjoin_nearest(
    unmatched_lines,
    segs_lines,
    how="left",
    max_distance=MATCH_DIST_M,
    distance_col="match_dist"
)
matched_p2 = m2.dropna(subset=["seg_idx"]).copy()
print(f"  Pass 2 (line-to-line ≤{MATCH_DIST_M}m): {len(matched_p2):,} additional matched")

# Combine both passes
matched = pd.concat([
    matched_p1[["volume","voc","speed","seg_idx","match_dist"]],
    matched_p2[["volume","voc","speed","seg_idx","match_dist"]],
], ignore_index=True)
matched = matched.dropna(subset=["seg_idx"])
matched["seg_idx"] = matched["seg_idx"].astype(int)

# Aggregate (some segs may match multiple TPO links — take max volume)
vol_agg = matched.groupby("seg_idx").agg(
    volume=("volume","max"),
    voc=("voc","max"),
    tpo_speed=("speed","mean"),
).reset_index()

segs = segs.merge(vol_agg, on="seg_idx", how="left")
segs["has_volume"] = segs["volume"].notna().astype(int)
segs["volume"]     = segs["volume"].fillna(0)
segs["voc"]        = segs["voc"].fillna(0)
segs["tpo_speed"]  = segs["tpo_speed"].fillna(segs["speed_kph"])

n_matched = segs["has_volume"].sum()
print(f"  TPO links: {len(assign):,}")
print(f"  Overture segments matched: {n_matched:,}")
print(f"  Volume range (matched): {segs[segs['has_volume']==1]['volume'].min():.0f} – "
      f"{segs[segs['has_volume']==1]['volume'].max():.0f}")

# =============================================================================
# Section 6: Compute criticality labels
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 6: Criticality labeling")
print("=" * 60)

labeled = segs[segs["has_volume"] == 1].copy()

# Normalise volume and betweenness to [0,1]
v_norm = (labeled["volume"] - labeled["volume"].min()) / \
         (labeled["volume"].max() - labeled["volume"].min() + 1e-9)
b_norm = (labeled["betweenness"] - labeled["betweenness"].min()) / \
         (labeled["betweenness"].max() - labeled["betweenness"].min() + 1e-9)

labeled["criticality_score"] = ALPHA * v_norm.values + (1 - ALPHA) * b_norm.values

threshold = np.percentile(labeled["criticality_score"], CRIT_PCTILE)
labeled["critical"] = (labeled["criticality_score"] >= threshold).astype(int)

print(f"  Labeled segments: {len(labeled):,}")
print(f"  Critical threshold (p{CRIT_PCTILE}): {threshold:.4f}")
print(f"  Critical:     {labeled['critical'].sum():,}  ({labeled['critical'].mean()*100:.1f}%)")
print(f"  Non-critical: {(1-labeled['critical']).sum():,}")

# Save scores back to full segs (for inference output)
segs = segs.merge(
    labeled[["seg_idx","criticality_score","critical"]],
    on="seg_idx", how="left"
)

# =============================================================================
# Section 6b: Spatial-join TAZ morphology features onto segments
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 6b: TAZ morphology features")
print("=" * 60)

# TAZ morphology features give the GNN demand-side context (land use, buildings,
# employment) in addition to the road-supply topology it already has.
# NOTE: volume/V-C ratio are intentionally NOT added as node features because
# the 61k unlabeled segments have no volume data — the model must learn from
# topology+morphology alone and generalize to unlabeled segments.

TAZ_FEAT_COLS = ["TOTAL_EMP","HH","building_coverage_pct",
                 "street_density_km_km2","building_density_n_km2","avg_footprint_m2"]

try:
    # zones_clean.gpkg already contains ALL morphology columns — no separate CSV needed
    zones_geo = gpd.read_file(OUT / "zones_clean.gpkg").to_crs(TARGET_CRS)

    avail_taz = [c for c in TAZ_FEAT_COLS if c in zones_geo.columns]
    print(f"  TAZ feature columns found in zones_clean.gpkg: {avail_taz}")

    # Spatial-join: segment centroid → TAZ polygon it falls inside
    seg_cents = segs[["seg_idx","geometry"]].copy()
    seg_cents["geometry"] = seg_cents.geometry.centroid

    taz_join = gpd.sjoin(
        seg_cents,
        zones_geo[["geometry"] + avail_taz],
        how="left", predicate="within"
    )
    # Drop duplicates (centroid exactly on a shared boundary)
    taz_join = taz_join[~taz_join.index.duplicated(keep="first")]

    for c in avail_taz:
        segs[c] = taz_join[c].reindex(segs.index).values

    # Fill the small fraction of segments outside all TAZ polygons with county median
    for c in avail_taz:
        segs[c] = segs[c].fillna(segs[c].median())

    n_matched_taz = taz_join[avail_taz[0]].notna().sum() if avail_taz else 0
    print(f"  Segments with TAZ match: {n_matched_taz:,} / {len(segs):,}")

except Exception as e:
    print(f"  WARNING: TAZ morphology join failed ({e}), proceeding without")
    avail_taz = []

# =============================================================================
# Section 7: Build node feature matrix
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 7: Node features")
print("=" * 60)

# Road-intrinsic features (available for ALL 65k segments)
BASE_FEATS = ["length_m","class_enc","connector_count","speed_kph",
              "has_surface","graph_degree","betweenness",
              "is_bridge","is_link","is_tunnel","is_private","sinuosity"]

# TAZ morphology features (land use / built environment context)
MORPH_FEATS = [c for c in TAZ_FEAT_COLS if c in segs.columns]

FEAT_COLS = BASE_FEATS + MORPH_FEATS

X_raw = segs[FEAT_COLS].fillna(0).values.astype(np.float32)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_raw).astype(np.float32)

print(f"  Road-intrinsic features ({len(BASE_FEATS)}): {BASE_FEATS}")
print(f"  TAZ morphology features ({len(MORPH_FEATS)}): {MORPH_FEATS}")
print(f"  Feature matrix total: {X_norm.shape}")

# =============================================================================
# Section 8: PyG Data object
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 8: Build PyG graph")
print("=" * 60)

# Labels: only for matched/labeled segments (-1 = unlabeled)
y_all = torch.full((len(segs),), -1, dtype=torch.long)
labeled_mask_np = segs["critical"].notna().values
y_np = segs["critical"].fillna(-1).astype(int).values
y_all = torch.tensor(y_np, dtype=torch.long)

labeled_mask = torch.tensor(labeled_mask_np, dtype=torch.bool)

data = Data(
    x          = torch.tensor(X_norm),
    edge_index = edge_index,
    y          = y_all,
)
data.labeled_mask = labeled_mask

print(f"  Nodes: {data.num_nodes:,}  Edges: {data.num_edges:,}")
print(f"  Labeled nodes: {labeled_mask.sum().item():,}")
torch.save(data, CRIT_DIR / "road_graph.pt")

# =============================================================================
# Section 9: Spatial 5-fold CV on labeled segments
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 9: Spatial {} -fold CV".format(K_FOLDS))
print("=" * 60)

cx = segs.geometry.centroid.x.values
cy = segs.geometry.centroid.y.values
km = KMeans(n_clusters=K_FOLDS, random_state=42, n_init=10)
all_folds = km.fit_predict(np.column_stack([cx, cy]))

# Only labeled nodes participate in CV
labeled_idx = np.where(labeled_mask_np)[0]
labeled_segs = segs.iloc[labeled_idx].copy()
labeled_segs["fold"] = all_folds[labeled_idx]

print(f"  Fold sizes: { {k: int((labeled_segs['fold']==k).sum()) for k in range(K_FOLDS)} }")

# =============================================================================
# Section 10: GAT model
# =============================================================================

class RoadGAT(nn.Module):
    """
    Two-layer Graph Attention Network for binary node classification.
    GAT is preferred over SAGE here because attention weights reveal
    WHICH neighboring segments influence the criticality prediction.
    """
    def __init__(self, in_ch, hidden, heads=4, dropout=0.4):
        super().__init__()
        self.conv1  = GATConv(in_ch, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2  = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.head   = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return self.head(x)


def train_epoch(model, opt, data, train_mask):
    model.train()
    opt.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def evaluate_clf(model, data, mask):
    model.eval()
    out   = model(data.x, data.edge_index)
    probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
    preds = out[mask].argmax(dim=1).cpu().numpy()
    truth = data.y[mask].cpu().numpy()
    loss  = F.cross_entropy(out[mask], data.y[mask]).item()
    return loss, probs, preds, truth


# =============================================================================
# Section 11: Training loop
# =============================================================================
cv_rows = []
best_fold_k    = None
best_fold_auc  = -1
best_fold_data = None

for fold in range(K_FOLDS):
    test_labeled_idx  = labeled_idx[labeled_segs["fold"].values == fold]
    train_labeled_idx = labeled_idx[labeled_segs["fold"].values != fold]

    train_mask = torch.zeros(len(segs), dtype=torch.bool)
    test_mask  = torch.zeros(len(segs), dtype=torch.bool)
    train_mask[train_labeled_idx] = True
    test_mask[test_labeled_idx]   = True

    # Check both classes represented in training
    y_train = data.y[train_mask].numpy()
    if len(np.unique(y_train)) < 2:
        print(f"  Fold {fold}: skipping — only one class in training set")
        continue

    model = RoadGAT(len(FEAT_COLS), HIDDEN, HEADS, DROPOUT)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)

    best_val = np.inf
    best_wts = None
    patience_cnt = 0
    tr_losses, va_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_l = train_epoch(model, opt, data, train_mask)
        va_l, _, _, _ = evaluate_clf(model, data, test_mask)
        sched.step(va_l)
        tr_losses.append(tr_l)
        va_losses.append(va_l)

        if va_l < best_val - 1e-5:
            best_val = va_l
            best_wts = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Fold {fold}: early stop epoch {epoch}")
                break

    model.load_state_dict(best_wts)
    torch.save(best_wts, CRIT_DIR / f"gat_fold{fold}.pt")

    _, probs, preds, truth = evaluate_clf(model, data, test_mask)
    auc = roc_auc_score(truth, probs)
    f1  = f1_score(truth, preds, zero_division=0)
    pre = precision_score(truth, preds, zero_division=0)
    rec = recall_score(truth, preds, zero_division=0)

    print(f"  Fold {fold}  n_test={test_mask.sum().item():4d}  "
          f"AUC={auc:.3f}  F1={f1:.3f}  Prec={pre:.3f}  Rec={rec:.3f}")

    cv_rows.append({"fold": fold, "AUC": auc, "F1": f1,
                    "Precision": pre, "Recall": rec,
                    "n_test": test_mask.sum().item(),
                    "tr_losses": tr_losses, "va_losses": va_losses})

    if auc > best_fold_auc:
        best_fold_auc  = auc
        best_fold_k    = fold
        best_fold_data = {"probs": probs, "preds": preds, "truth": truth,
                          "tr": tr_losses, "va": va_losses,
                          "test_mask": test_mask.numpy()}

cv_metric_df = pd.DataFrame([{k: v for k, v in r.items()
                               if k not in ["tr_losses","va_losses"]}
                              for r in cv_rows])
cv_metric_df.to_csv(CRIT_DIR / "cv_criticality_results.csv", index=False)

print(f"\n  CV Summary (mean):")
print(cv_metric_df[["AUC","F1","Precision","Recall"]].mean().round(3).to_string())

# =============================================================================
# Section 12: Full inference on all segments
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 12: Full inference → all segments")
print("=" * 60)

# Train final model on ALL labeled data
final_model = RoadGAT(len(FEAT_COLS), HIDDEN, HEADS, DROPOUT)
final_opt   = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)

best_val = np.inf
best_wts = None
patience_cnt = 0
for epoch in range(1, EPOCHS + 1):
    tr_l = train_epoch(final_model, final_opt, data, labeled_mask)
    if tr_l < best_val - 1e-5:
        best_val = tr_l
        best_wts = copy.deepcopy(final_model.state_dict())
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Final model: early stop epoch {epoch}")
            break

final_model.load_state_dict(best_wts)
torch.save(best_wts, CRIT_DIR / "gat_final.pt")

with torch.no_grad():
    final_model.eval()
    all_out   = final_model(data.x, data.edge_index)
    all_probs = F.softmax(all_out, dim=1)[:, 1].numpy()
    all_preds = all_out.argmax(dim=1).numpy()

segs["pred_prob_critical"]  = all_probs
segs["pred_critical"]       = all_preds

# Export results
out_cols = ["seg_idx","class","length_m","connector_count","speed_kph",
            "graph_degree","betweenness","volume","voc","criticality_score",
            "critical","pred_prob_critical","pred_critical","geometry"]
segs_out = gpd.GeoDataFrame(segs[[c for c in out_cols if c in segs.columns]],
                             geometry="geometry", crs=TARGET_CRS)
segs_out.to_file(CRIT_DIR / "critical_segments.gpkg", driver="GPKG")
segs_out.drop(columns="geometry").to_csv(CRIT_DIR / "criticality_scores.csv", index=False)

n_pred_critical = int(all_preds.sum())
print(f"  Segments predicted critical: {n_pred_critical:,} / {len(segs):,} "
      f"({n_pred_critical/len(segs)*100:.1f}%)")

# =============================================================================
# Section 13: Figures
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 13: Figures")
print("=" * 60)

# --- Fig C1: Score distribution ---
print("  Fig C1: Score distribution...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.hist(labeled["criticality_score"], bins=40, color="#2C3E50", alpha=0.75, edgecolor="white")
ax.axvline(threshold, color="red", linewidth=1.5, linestyle="--",
           label=f"Critical threshold\n(p{CRIT_PCTILE} = {threshold:.3f})")
ax.set_xlabel("Criticality score", fontsize=10)
ax.set_ylabel("Segment count", fontsize=10)
ax.set_title("Combined Criticality Score\n(0.5×volume + 0.5×betweenness)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(labeled["volume"], bins=40, color="#E74C3C", alpha=0.75, edgecolor="white")
ax.set_xlabel("TPO assigned volume (trips/day)", fontsize=10)
ax.set_title("TPO Volume Distribution\n(matched segments)", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

ax = axes[2]
ax.hist(labeled["betweenness"], bins=40, color="#3498DB", alpha=0.75, edgecolor="white")
ax.set_xlabel("Betweenness centrality", fontsize=10)
ax.set_title("Betweenness Centrality Distribution\n(driveable road graph)", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

fig.suptitle("Knox County Road Criticality — Score Components", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figC1_criticality_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figC1_criticality_score_distribution.png")

# --- Fig C2: Criticality map ---
print("  Fig C2: Criticality map...")
zones = gpd.read_file(OUT / "zones_reprojected.gpkg")
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Left: ground truth critical roads (labeled only)
ax = axes[0]
zones.plot(ax=ax, color="#f5f5f5", edgecolor="#dddddd", linewidth=0.2, zorder=1)
nc = segs_out[(segs_out["critical"]==0) & segs_out["critical"].notna()]
ct = segs_out[(segs_out["critical"]==1) & segs_out["critical"].notna()]
nc.plot(ax=ax, color="#AAAAAA", linewidth=0.5, alpha=0.5, zorder=2, label="Non-critical")
ct.plot(ax=ax, color="#E74C3C", linewidth=1.2, alpha=0.9, zorder=3, label="Critical (ground truth)")
ax.set_title(f"Ground Truth Critical Roads\n(TPO-labeled, top {100-CRIT_PCTILE}% combined score)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_axis_off()

# Right: GNN predicted criticality on ALL segments
ax = axes[1]
zones.plot(ax=ax, color="#f5f5f5", edgecolor="#dddddd", linewidth=0.2, zorder=1)
nc_pred = segs_out[segs_out["pred_critical"] == 0]
ct_pred = segs_out[segs_out["pred_critical"] == 1]
nc_pred.plot(ax=ax, color="#CCCCCC", linewidth=0.3, alpha=0.4, zorder=2)
ct_pred.plot(ax=ax, color="#E74C3C", linewidth=1.0, alpha=0.85, zorder=3,
             label=f"Predicted critical ({len(ct_pred):,} segments)")
ax.set_title(f"GNN Predicted Critical Roads\n(all {len(segs):,} Overture segments)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_axis_off()

fig.suptitle("Knox County Road Criticality — Ground Truth vs GNN Prediction\n"
             "Red = critical (high volume + high betweenness)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figC2_criticality_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figC2_criticality_map.png")

# --- Fig C3: ROC curve (best fold) ---
print("  Fig C3: ROC curve...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
fpr, tpr, _ = roc_curve(best_fold_data["truth"], best_fold_data["probs"])
auc = roc_auc_score(best_fold_data["truth"], best_fold_data["probs"])
ax.plot(fpr, tpr, color="#E74C3C", linewidth=2, label=f"GAT  AUC={auc:.3f}")
ax.plot([0,1],[0,1],"k--",linewidth=1, label="Random")
ax.fill_between(fpr, tpr, alpha=0.15, color="#E74C3C")
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title(f"ROC Curve — GAT (Best Fold {best_fold_k})", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
cm = confusion_matrix(best_fold_data["truth"], best_fold_data["preds"])
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Predicted\nNon-critical","Predicted\nCritical"])
ax.set_yticklabels(["True\nNon-critical","True\nCritical"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i,j] > cm.max()/2 else "black")
ax.set_title(f"Confusion Matrix — Best Fold {best_fold_k}", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Knox County Road Criticality GNN — Evaluation", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figC3_gnn_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figC3_gnn_roc_curve.png")

# --- Fig C4: Feature importance (permutation test on best fold) ---
print("  Fig C4: Feature importance...")

# Simple permutation importance: shuffle each feature, measure AUC drop
base_auc = best_fold_auc
bfd = best_fold_data
test_mask_t = torch.tensor(bfd["test_mask"], dtype=torch.bool)
importances = []

final_model.eval()
with torch.no_grad():
    for i, feat in enumerate(FEAT_COLS):
        X_perm = data.x.clone()
        perm_idx = torch.randperm(X_perm.shape[0])
        X_perm[:, i] = X_perm[perm_idx, i]    # shuffle feature i
        out_p  = final_model(X_perm, data.edge_index)
        probs_p = F.softmax(out_p[test_mask_t], dim=1)[:,1].numpy()
        truth_t = data.y[test_mask_t].numpy()
        try:
            auc_p = roc_auc_score(truth_t, probs_p)
        except Exception:
            auc_p = 0.5
        importances.append({"feature": feat, "auc_drop": base_auc - auc_p})

imp_df = pd.DataFrame(importances).sort_values("auc_drop", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#E74C3C" if v > 0 else "#3498DB" for v in imp_df["auc_drop"]]
ax.barh(range(len(imp_df)), imp_df["auc_drop"].values, color=colors, alpha=0.8, edgecolor="white")
ax.set_yticks(range(len(imp_df)))
ax.set_yticklabels(imp_df["feature"].values, fontsize=10)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("AUC drop when feature is shuffled\n(higher = more important)", fontsize=10)
ax.set_title("GAT Feature Importance\n(Permutation — Red = positive contribution)", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "figC4_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figC4_feature_importance.png")

# =============================================================================
# Final summary
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 6 COMPLETE")
print("=" * 60)
print(f"""
  Labeled segments (TPO-matched):  {n_matched:,}
  Critical (ground truth):         {int(labeled['critical'].sum()):,}
  Predicted critical (all segs):   {n_pred_critical:,}

  GAT 5-fold CV results (mean):
{cv_metric_df[['AUC','F1','Precision','Recall']].mean().round(3).to_string()}

  Outputs:
    critical_segments.gpkg     — all segments with criticality scores
    criticality_scores.csv     — full table
    gat_final.pt               — trained model weights
    cv_criticality_results.csv — per-fold metrics

  Key:
    AUC > 0.8  → strong prediction from topology alone
    AUC 0.7–0.8 → moderate — topology partially explains criticality
    AUC < 0.7  → road class + volume dominate, topology adds little

  Figures: figC1–C4
""")
