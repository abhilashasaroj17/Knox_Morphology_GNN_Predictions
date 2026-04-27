"""
Knox County City2Graph - Step 5: GNN Training
==============================================
Model: Heterogeneous GraphSAGE — SPARSE edge variant
  Message-passing channels:
    1. adjacent  edges  (spatial contiguity, ~3k edges)
    2. top_od    edges  (top-10 OD flows per origin zone, ~5k edges)
  OD full matrix NOT used for message passing (would cause over-smoothing
  on this near-complete 508-node graph). Instead, per-zone OD statistics
  (log in-strength, out-strength, net-flow) are added as node features.

Architecture:
  ResidualSAGE(
    Linear(11 → 64) input projection + skip
    SAGEConv(64 → 64) x2 edge types → LayerNorm → ReLU → Dropout
    SAGEConv(64 → 64) x2 edge types → LayerNorm → ReLU → Dropout
    Linear(64 → 2)
  )

Training:
  - 5-fold spatial CV (masks from graph metadata)
  - Loss: MSE on log-scale targets
  - Optimizer: Adam lr=5e-4, weight_decay=1e-4
  - Epochs: 500, early stopping patience=50
  - Baseline: OLS & Poisson from Step 3

Outputs (outputs/gnn/):
  - cv_gnn_results.csv
  - figG5_training_curves.png
  - figG6_gnn_vs_regression.png
  - figG7_gnn_predictions.png
  - figG8_gnn_residual_maps.png
  - model_fold{k}.pt

Run with:
    $env:PYTHONUTF8="1"; .venv\\Scripts\\Activate.ps1; python step5_gnn_train.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import json, copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# Paths & config
# =============================================================================
ROOT    = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph")
OUT     = ROOT / "outputs"
GNN_DIR = OUT / "gnn"
FIG_DIR = OUT / "figures"
GNN_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device("cpu")
EPOCHS   = 500
LR       = 5e-4
WD       = 1e-4
HIDDEN   = 64
DROPOUT  = 0.3
PATIENCE = 50
TOP_K    = 10     # top-K OD destinations per origin zone

# =============================================================================
# Section 1: Load graph + build enriched node features
# =============================================================================
print("=" * 60)
print("  SECTION 1: Load graph")
print("=" * 60)

data = torch.load(OUT / "knox_hetero_graph.pt", weights_only=False)
meta = json.loads((OUT / "graph_metadata.json").read_text())
od_all = pd.read_csv(OUT / "od_long_2026_internal.csv")

N          = data["zone"].x.shape[0]
fold_masks = data["zone"].fold_masks
K          = meta["n_folds"]
zone_ids   = data["zone"].zone_ids.numpy()
zone_to_idx = {int(z): i for i, z in enumerate(zone_ids)}

# --- Build top-K OD edges per origin (sparse, avoids over-smoothing) ---
od_valid = od_all[
    od_all["origin"].isin(zone_to_idx) & od_all["destination"].isin(zone_to_idx)
].copy()
od_valid["src"] = od_valid["origin"].map(zone_to_idx).astype(int)
od_valid["dst"] = od_valid["destination"].map(zone_to_idx).astype(int)
od_valid = od_valid[od_valid["src"] != od_valid["dst"]]
topk_od = (od_valid.sort_values("flow", ascending=False)
                   .groupby("src", sort=False)
                   .head(TOP_K)
                   .reset_index(drop=True)
                   [["src","dst","flow"]])
src_topk = torch.tensor(topk_od["src"].values, dtype=torch.long)
dst_topk = torch.tensor(topk_od["dst"].values, dtype=torch.long)
edge_attr_topk = torch.tensor(
    np.log1p(topk_od["flow"].values).astype(np.float32)
).unsqueeze(1)
print(f"  Top-{TOP_K} OD edges per zone: {src_topk.shape[0]:,}")

# --- Add OD strength stats as extra node features ---
# (replaces using full OD as message-passing edges)
od_stats = pd.read_csv(OUT / "od_zone_stats.csv")
zone_df  = pd.DataFrame({"zone_id": zone_ids})
zone_df  = zone_df.merge(od_stats[["zone_id","production","attraction","net_flow"]],
                         on="zone_id", how="left").fillna(0)

scaler_od = StandardScaler()
od_feat = scaler_od.fit_transform(
    np.log1p(np.abs(zone_df[["production","attraction","net_flow"]].values))
).astype(np.float32)
# sign of net_flow as 4th feature
od_feat = np.hstack([od_feat,
    np.sign(zone_df["net_flow"].values).reshape(-1,1).astype(np.float32)])

# Concatenate onto existing 8 morphology+landuse features  → 12 total
X_full = np.hstack([
    data["zone"].x.numpy(),   # 8 normalised features from step4
    od_feat                    # 4 OD stats
]).astype(np.float32)
N_FEATS = X_full.shape[1]

print(f"  Nodes: {N}  Features: {N_FEATS} (8 morph/landuse + 4 OD stats)  Folds: {K}")
print(f"  Adjacent edges: {data['zone','adjacent','zone'].edge_index.shape[1]:,}")
print(f"  Top-{TOP_K} OD edges: {src_topk.shape[0]:,}")

# =============================================================================
# Section 2: Model definition (sparse edges, residual connections)
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 2: Model")
print("=" * 60)

class ZoneSAGE(nn.Module):
    """
    Two-layer GraphSAGE with residual skip connections and LayerNorm.
    Uses only sparse edges (adjacent + top-K OD) — avoids over-smoothing.
    """
    def __init__(self, in_channels, hidden, out_channels=2, dropout=0.3):
        super().__init__()
        self.proj   = nn.Linear(in_channels, hidden)    # input projection
        self.conv1  = SAGEConv(hidden, hidden)
        self.conv2  = SAGEConv(hidden, hidden)
        self.norm1  = nn.LayerNorm(hidden)
        self.norm2  = nn.LayerNorm(hidden)
        self.head   = nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.proj(x))
        # Layer 1 + residual
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h) + x                               # skip
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Layer 2 + residual
        h2 = self.conv2(h, edge_index)
        h2 = self.norm2(h2)
        h2 = F.relu(h2) + h                             # skip
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        return self.head(h2)


def build_hetero_model():
    """Sparse hetero model: adjacent + top-K OD channels only."""
    base = ZoneSAGE(N_FEATS, HIDDEN, out_channels=2, dropout=DROPOUT)
    metadata = (
        ["zone"],
        [("zone","adjacent","zone"),
         ("zone","topk_od","zone")],
    )
    return to_hetero(base, metadata=metadata, aggr="mean")


# Sanity check
_m  = build_hetero_model().to(DEVICE)
_x  = {"zone": torch.tensor(X_full).to(DEVICE)}
_ei = {
    ("zone","adjacent","zone"): data["zone","adjacent","zone"].edge_index.to(DEVICE),
    ("zone","topk_od","zone"):  torch.stack([src_topk, dst_topk]).to(DEVICE),
}
with torch.no_grad():
    _out = _m(_x, _ei)
print(f"  Model output shape: {_out['zone'].shape}  ✓")
n_params = sum(p.numel() for p in _m.parameters())
print(f"  Parameters: {n_params:,}")
del _m, _out

# =============================================================================
# Section 3: Training loop helpers
# =============================================================================

def train_one_epoch(model, optimizer, x_dict, ei_dict, y, train_mask):
    model.train()
    optimizer.zero_grad()
    out  = model(x_dict, ei_dict)["zone"]
    loss = F.mse_loss(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, x_dict, ei_dict, y, mask):
    model.eval()
    out  = model(x_dict, ei_dict)["zone"]
    loss = F.mse_loss(out[mask], y[mask]).item()
    return loss, out[mask].cpu().numpy()


def metrics_original_scale(y_log_true, y_log_pred):
    """Convert log-scale preds back to trip counts, compute metrics."""
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    y_pred = np.clip(y_pred, 0, None)
    results = {}
    for i, col in enumerate(["production", "attraction"]):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        results[col] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return results, y_true, y_pred

# =============================================================================
# Section 4: 5-Fold spatial CV training
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 4: 5-Fold Spatial CV Training")
print("=" * 60)

# Move data to device once — use enriched features + sparse edges
x_dict = {"zone": torch.tensor(X_full).to(DEVICE)}
ei_dict = {
    ("zone","adjacent","zone"): data["zone","adjacent","zone"].edge_index.to(DEVICE),
    ("zone","topk_od","zone"):  torch.stack([src_topk, dst_topk]).to(DEVICE),
}
y_all = data["zone"].y.to(DEVICE)   # log-scale

cv_rows   = []
fold_curves = {}
best_fold_k = None
best_fold_r2 = -np.inf
best_fold_preds = None
best_fold_true  = None

for fold in range(K):
    train_mask = fold_masks[fold]["train"].to(DEVICE)
    test_mask  = fold_masks[fold]["test"].to(DEVICE)
    n_train    = train_mask.sum().item()
    n_test     = test_mask.sum().item()

    model     = build_hetero_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-5)

    best_val_loss = np.inf
    best_weights  = None
    patience_cnt  = 0
    train_losses  = []
    val_losses    = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, optimizer, x_dict, ei_dict, y_all, train_mask)
        va_loss, _ = evaluate(model, x_dict, ei_dict, y_all, test_mask)
        scheduler.step(va_loss)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_weights  = copy.deepcopy(model.state_dict())
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Fold {fold}: early stop at epoch {epoch}")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    torch.save(best_weights, GNN_DIR / f"model_fold{fold}.pt")

    # Final evaluation
    _, test_preds = evaluate(model, x_dict, ei_dict, y_all, test_mask)
    y_log_true = y_all[test_mask].cpu().numpy()
    metrics, y_true_raw, y_pred_raw = metrics_original_scale(y_log_true, test_preds)

    # Baseline: predict training mean for each target
    y_train_log = y_all[train_mask].cpu().numpy()
    train_mean  = y_train_log.mean(axis=0)
    baseline_pred = np.tile(train_mean, (n_test, 1))
    base_metrics, _, _ = metrics_original_scale(y_log_true, baseline_pred)

    fold_curves[fold] = {"train": train_losses, "val": val_losses}

    avg_r2 = (metrics["production"]["R2"] + metrics["attraction"]["R2"]) / 2
    print(f"  Fold {fold}  n_test={n_test:3d}  "
          f"prod  MAE={metrics['production']['MAE']:.0f}  R²={metrics['production']['R2']:.3f}  | "
          f"attr  MAE={metrics['attraction']['MAE']:.0f}  R²={metrics['attraction']['R2']:.3f}")

    if avg_r2 > best_fold_r2:
        best_fold_r2    = avg_r2
        best_fold_k     = fold
        best_fold_preds = y_pred_raw
        best_fold_true  = y_true_raw
        best_fold_mask  = test_mask.cpu().numpy()

    for col in ["production", "attraction"]:
        cv_rows.append({
            "fold": fold, "target": col, "model": "GNN_SAGE",
            "MAE":  metrics[col]["MAE"],
            "RMSE": metrics[col]["RMSE"],
            "R2":   metrics[col]["R2"],
            "baseline_MAE":  base_metrics[col]["MAE"],
            "baseline_RMSE": base_metrics[col]["RMSE"],
            "n_test": n_test,
        })

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(GNN_DIR / "cv_gnn_results.csv", index=False)

print("\n  GNN CV summary (mean across folds):")
summary = cv_df.groupby(["model","target"])[["MAE","RMSE","R2"]].mean().round(2)
print(summary.to_string())

# =============================================================================
# Section 5: Load Step 3 regression CV results for comparison
# =============================================================================
reg_cv_path = OUT / "regression" / "cv_results.csv"
reg_cv = pd.read_csv(reg_cv_path) if reg_cv_path.exists() else None

# =============================================================================
# Section 6: Figures
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 6: Figures")
print("=" * 60)

# --- Fig G5: Training curves ---
print("  Fig G5: Training curves...")
fig, axes = plt.subplots(1, K, figsize=(20, 4), sharey=False)
for fold in range(K):
    ax = axes[fold]
    tr = fold_curves[fold]["train"]
    va = fold_curves[fold]["val"]
    ax.plot(tr, color="#E74C3C", linewidth=1.2, alpha=0.8, label="Train MSE")
    ax.plot(va, color="#3498DB", linewidth=1.2, alpha=0.8, label="Val MSE")
    ax.set_title(f"Fold {fold}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("MSE (log scale)" if fold == 0 else "", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if fold == best_fold_k:
        ax.set_facecolor("#FFFDE7")
        ax.set_title(f"Fold {fold}  ★ best", fontsize=10, fontweight="bold")
fig.suptitle("GNN Training Curves — 5 Spatial Folds\n"
             "Red = train MSE  |  Blue = validation MSE  |  Yellow = best fold",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figG5_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figG5_training_curves.png")

# --- Fig G6: GNN vs Regression comparison ---
print("  Fig G6: GNN vs regression...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
metrics_list = ["MAE", "RMSE", "R2"]
metric_labels = {"MAE": "MAE (trips)", "RMSE": "RMSE (trips)", "R2": "R²"}

gnn_summary = cv_df.groupby("target")[["MAE","RMSE","R2"]].agg(["mean","std"])

for ax, met in zip(axes, metrics_list):
    x_positions = np.arange(2)   # 0=production, 1=attraction
    bar_w = 0.22
    model_specs = [("GNN_SAGE", "#E74C3C", cv_df)]
    if reg_cv is not None:
        model_specs += [
            ("OLS",     "#3498DB", reg_cv[reg_cv["model"] == "OLS"]),
            ("Poisson", "#2ECC71", reg_cv[reg_cv["model"] == "Poisson"]),
        ]

    for mi, (mname, color, df_src) in enumerate(model_specs):
        for ti, target in enumerate(["production", "attraction"]):
            vals = df_src[df_src["target"] == target][met].dropna()
            xp   = ti + (mi - 1) * bar_w
            ax.bar(xp, vals.mean(), width=bar_w, color=color,
                   alpha=0.85, edgecolor="white",
                   label=mname if ti == 0 else None)
            ax.errorbar(xp, vals.mean(), yerr=vals.std(),
                        fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Production", "Attraction"], fontsize=10)
    ax.set_ylabel(metric_labels[met], fontsize=10)
    ax.set_title(f"5-fold CV: {metric_labels[met]}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    if ax == axes[2]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)

fig.suptitle("Knox County — GNN vs Regression Baselines\n"
             "Spatial 5-fold CV  |  Error bars = ±1 std",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figG6_gnn_vs_regression.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figG6_gnn_vs_regression.png")

# --- Fig G7: Predicted vs Actual (best fold) ---
print("  Fig G7: Predicted vs actual (best fold)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
target_names = ["Production", "Attraction"]
for ax, i, tname in zip(axes, range(2), target_names):
    yt = best_fold_true[:, i]
    yp = best_fold_preds[:, i]
    ax.scatter(yt, yp, s=18, alpha=0.55, color="#2C3E50", edgecolors="none")
    lim = max(yt.max(), yp.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect")
    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    ax.set_title(f"GNN — {tname}\nR²={r2:.3f}  RMSE={rmse:.0f}  MAE={mae:.0f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual trips", fontsize=10)
    ax.set_ylabel("Predicted trips", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

fig.suptitle(f"GNN Predicted vs Actual — Best Fold (fold {best_fold_k})\n"
             "All values in original trip count scale",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figG7_gnn_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figG7_gnn_predictions.png")

# --- Fig G8: Spatial residual maps (best fold) ---
print("  Fig G8: Spatial residual maps (best fold)...")
zones_geo = gpd.read_file(OUT / "zones_clean.gpkg")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, i, tname in zip(axes, range(2), ["Production", "Attraction"]):
    # Build a residual series aligned to all zones (NaN for training zones)
    resid_series = pd.Series(np.nan, index=range(N))
    test_indices = np.where(best_fold_mask)[0]
    resid_vals   = best_fold_true[:, i] - best_fold_preds[:, i]
    resid_series.iloc[test_indices] = resid_vals

    zones_geo["resid"] = resid_series.values
    vmax = np.nanpercentile(np.abs(resid_series.dropna()), 95)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    zones_geo.plot(ax=ax, color="#eeeeee", edgecolor="#cccccc", linewidth=0.2, zorder=1)
    test_gdf = zones_geo[~zones_geo["resid"].isna()].copy()
    test_gdf.plot(column="resid", ax=ax, cmap="RdBu_r", norm=norm,
                  legend=True,
                  legend_kwds={"label": "Residual (trips)", "shrink": 0.6},
                  edgecolor="white", linewidth=0.2, zorder=2)
    ax.set_title(f"GNN Residuals — {tname}\n"
                 "Grey = training zones  |  Red = over-predicted  |  Blue = under-predicted",
                 fontsize=11, fontweight="bold")
    ax.set_axis_off()

fig.suptitle(f"Knox County GNN: Spatial Residuals (Best Fold {best_fold_k})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figG8_gnn_residual_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figG8_gnn_residual_maps.png")

# =============================================================================
# Final summary
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 5 COMPLETE")
print("=" * 60)

gnn_mean = cv_df.groupby("target")[["MAE","RMSE","R2"]].mean()
print("\n  GNN (GraphSAGE) — mean across 5 spatial folds:")
print(gnn_mean.round(2).to_string())

if reg_cv is not None:
    ols_mean  = reg_cv[reg_cv["model"]=="OLS"].groupby("target")[["MAE","RMSE","R2"]].mean()
    pois_mean = reg_cv[reg_cv["model"]=="Poisson"].groupby("target")[["MAE","RMSE","R2"]].mean()
    print("\n  OLS — mean across 5 spatial folds:")
    print(ols_mean.round(2).to_string())
    print("\n  Poisson — mean across 5 spatial folds:")
    print(pois_mean.round(2).to_string())

    gnn_r2_prod  = gnn_mean.loc["production", "R2"]
    ols_r2_prod  = ols_mean.loc["production", "R2"]
    improvement  = gnn_r2_prod - ols_r2_prod
    print(f"\n  R² improvement GNN vs OLS (production): {improvement:+.3f}")

print(f"\n  Best fold: {best_fold_k}  (avg R²={best_fold_r2:.3f})")
print(f"\n  Figures saved to:  {FIG_DIR}")
print(f"  Models saved to:   {GNN_DIR}")
print(f"  CV results:        {GNN_DIR / 'cv_gnn_results.csv'}")
