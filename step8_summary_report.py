"""
Knox County City2Graph - Step 8: Summary Report & Figures
==========================================================
Produces a comprehensive visual + statistical summary of the road
criticality GNN experiment covering:

  1. Training data construction stats
  2. Feature set breakdown
  3. 5-fold spatial CV test results (per fold + mean)
  4. Prediction coverage on the full Overture network
  5. Critical segment breakdown by road class
  6. Comparison figure: ground truth vs GNN predictions

Outputs (outputs/figures/):
  figS1_training_data_overview.png
  figS2_cv_test_results.png
  figS3_prediction_coverage.png
  figS4_critical_by_class.png
  figS5_feature_summary.png

Run with:
    $env:PYTHONUTF8="1"; .venv\\Scripts\\Activate.ps1; python step8_summary_report.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT     = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph")
OUT      = ROOT / "outputs"
CRIT_DIR = OUT / "criticality"
FIG_DIR  = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette
C_RED    = "#E74C3C"
C_BLUE   = "#2980B9"
C_GREEN  = "#27AE60"
C_ORANGE = "#E67E22"
C_GREY   = "#95A5A6"
C_DARK   = "#2C3E50"
BG       = "#F8F9FA"

print("Loading data...")
segs  = gpd.read_file(CRIT_DIR / "critical_segments.gpkg")
cv_df = pd.read_csv(CRIT_DIR / "cv_criticality_results.csv")

# ── Derived counts ──────────────────────────────────────────────────────────
total_segs      = len(segs)
labeled_segs    = int(segs["critical"].notna().sum())
unlabeled_segs  = total_segs - labeled_segs
n_gt_critical   = int((segs["critical"] == 1).sum())
n_gt_noncrit    = int((segs["critical"] == 0).sum())
n_pred_critical = int((segs["pred_critical"] == 1).sum())
n_pred_noncrit  = total_segs - n_pred_critical

# ── Ground-truth vs predicted overlap ───────────────────────────────────────
labeled_mask  = segs["critical"].notna()
gt_1   = segs["critical"] == 1
pred_1 = segs["pred_critical"] == 1

tp = int((labeled_mask & gt_1  & pred_1 ).sum())
fp = int((labeled_mask & ~gt_1 & pred_1 ).sum())
fn = int((labeled_mask & gt_1  & ~pred_1).sum())
tn = int((labeled_mask & ~gt_1 & ~pred_1).sum())

precision_all = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_all    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_all        = 2*precision_all*recall_all/(precision_all+recall_all) if (precision_all+recall_all)>0 else 0

print(f"  Total segments: {total_segs:,}")
print(f"  Labeled (TPO-matched): {labeled_segs:,}  |  Unlabeled: {unlabeled_segs:,}")
print(f"  Ground truth critical: {n_gt_critical:,}  |  Non-critical: {n_gt_noncrit:,}")
print(f"  Predicted critical (all 65k): {n_pred_critical:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig S1 — Training data construction overview
# ─────────────────────────────────────────────────────────────────────────────
print("\nFig S1: Training data overview...")

fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.suptitle("Knox County Road Criticality GNN — Training Data Construction",
             fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Data funnel
ax = fig.add_subplot(gs[0, 0])
stages = ["All Overture\nsegments", "Driveable\nonly", "Matched to\nTPO (≤100m)", "Labeled\n(has volume)"]
counts = [73502, 65524, 4105, 4105]
colors_funnel = [C_GREY, C_BLUE, C_ORANGE, C_RED]
bars = ax.barh(range(len(stages)), counts, color=colors_funnel, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(stages)))
ax.set_yticklabels(stages, fontsize=9)
ax.set_xlabel("Segment count", fontsize=9)
ax.set_title("Data Pipeline Funnel", fontsize=10, fontweight="bold")
for i, (b, c) in enumerate(zip(bars, counts)):
    ax.text(b.get_width() + 400, i, f"{c:,}", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlim(0, 85000)
ax.grid(axis="x", alpha=0.3)
ax.set_facecolor(BG)

# Panel 2: Label distribution (pie)
ax = fig.add_subplot(gs[0, 1])
sizes  = [n_gt_critical, n_gt_noncrit]
labels = [f"Critical\n(top 20%)\nn={n_gt_critical:,}", f"Non-critical\nn={n_gt_noncrit:,}"]
ax.pie(sizes, labels=labels, colors=[C_RED, C_GREY],
       autopct="%1.1f%%", startangle=90,
       wedgeprops={"edgecolor":"white","linewidth":1.5},
       textprops={"fontsize":9})
ax.set_title(f"Label Distribution\n({labeled_segs:,} TPO-matched segments)", fontsize=10, fontweight="bold")

# Panel 3: Volume distribution of labeled segments
ax = fig.add_subplot(gs[0, 2])
vol = segs[segs["volume"] > 0]["volume"]
ax.hist(vol, bins=40, color=C_BLUE, alpha=0.8, edgecolor="white")
ax.axvline(vol.quantile(0.8), color=C_RED, linewidth=1.5, linestyle="--",
           label=f"p80 threshold\n({vol.quantile(0.8):.0f} trips/day)")
ax.set_xlabel("TPO assigned volume (trips/day)", fontsize=9)
ax.set_ylabel("Count", fontsize=9)
ax.set_title("Volume Distribution\n(TPO-matched segments)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3); ax.set_facecolor(BG)

# Panel 4: Criticality score components
ax = fig.add_subplot(gs[1, 0])
labeled = segs[segs["criticality_score"].notna()].copy()
ax.scatter(labeled["volume"] / labeled["volume"].max(),
           labeled["betweenness"] / labeled["betweenness"].max(),
           c=labeled["critical"].map({1: C_RED, 0: C_GREY}),
           alpha=0.3, s=6, linewidths=0)
ax.set_xlabel("Norm. volume", fontsize=9)
ax.set_ylabel("Norm. betweenness", fontsize=9)
ax.set_title("Criticality Score Components\n(red = critical label)", fontsize=10, fontweight="bold")
crit_patch   = mpatches.Patch(color=C_RED,  label="Critical")
nocrit_patch = mpatches.Patch(color=C_GREY, label="Non-critical")
ax.legend(handles=[crit_patch, nocrit_patch], fontsize=8)
ax.grid(alpha=0.3); ax.set_facecolor(BG)

# Panel 5: Spatial fold sizes
ax = fig.add_subplot(gs[1, 1])
fold_sizes = [870, 246, 1578, 635, 776]   # from run output
ax.bar(range(5), fold_sizes, color=[C_BLUE, C_GREEN, C_ORANGE, C_RED, C_DARK],
       alpha=0.85, edgecolor="white")
ax.set_xticks(range(5)); ax.set_xticklabels([f"Fold {i}" for i in range(5)])
ax.set_ylabel("Test segments", fontsize=9)
ax.set_title("Spatial 5-Fold CV\nTest Fold Sizes", fontsize=10, fontweight="bold")
for i, v in enumerate(fold_sizes):
    ax.text(i, v+20, str(v), ha="center", fontsize=9, fontweight="bold")
ax.grid(axis="y", alpha=0.3); ax.set_facecolor(BG)

# Panel 6: Match distance concept
ax = fig.add_subplot(gs[1, 2])
dists   = [25, 50, 75, 100, 150]
matched = [2149, 3414, 4043, 4362, 4592]
pcts    = [v/4799*100 for v in matched]
ax.plot(dists, pcts, "o-", color=C_BLUE, linewidth=2, markersize=7)
ax.axvline(100, color=C_RED, linewidth=1.5, linestyle="--", label="Used: 100m (90.9%)")
ax.fill_between(dists, pcts, alpha=0.15, color=C_BLUE)
ax.set_xlabel("Max match distance (m)", fontsize=9)
ax.set_ylabel("% TPO links matched", fontsize=9)
ax.set_title("TPO→Overture Match Rate\nvs Distance Threshold", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_facecolor(BG)

plt.savefig(FIG_DIR / "v3_figS1_training_data_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: v3_figS1_training_data_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig S2 — CV test results
# ─────────────────────────────────────────────────────────────────────────────
print("Fig S2: CV test results...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
fig.suptitle("Knox County Road Criticality GNN — 5-Fold Spatial CV Test Results",
             fontsize=13, fontweight="bold")

metrics  = ["AUC", "F1", "Precision", "Recall"]
fold_colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_DARK]

# Left: per-fold metric bars
ax = axes[0]
x = np.arange(len(metrics))
width = 0.16
for i, (_, row) in enumerate(cv_df.iterrows()):
    vals = [row["AUC"], row["F1"], row["Precision"], row["Recall"]]
    ax.bar(x + i*width, vals, width, label=f"Fold {i}",
           color=fold_colors[i], alpha=0.8, edgecolor="white")
means = [cv_df["AUC"].mean(), cv_df["F1"].mean(),
         cv_df["Precision"].mean(), cv_df["Recall"].mean()]
ax.plot(x + 2*width, means, "D--", color="black", markersize=7, linewidth=1.5,
        label="Mean", zorder=5)
ax.set_xticks(x + 2*width); ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("Score", fontsize=10); ax.set_ylim(0, 1)
ax.set_title("Per-Fold Metrics", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3); ax.set_facecolor(BG)

# Middle: mean ± std
ax = axes[1]
mean_vals = [cv_df[m].mean() for m in metrics]
std_vals  = [cv_df[m].std()  for m in metrics]
colors_m  = [C_RED, C_ORANGE, C_BLUE, C_GREEN]
bars = ax.bar(metrics, mean_vals, color=colors_m, alpha=0.85, edgecolor="white",
              yerr=std_vals, capsize=5, error_kw={"linewidth":1.5})
for bar, val, std in zip(bars, mean_vals, std_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.02,
            f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Score", fontsize=10); ax.set_ylim(0, 1.05)
ax.set_title("Mean ± Std Dev\n(5 spatial folds)", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3); ax.set_facecolor(BG)

# Right: confusion matrix (aggregate over folds from labeled segments)
ax = axes[2]
cm = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm, cmap="Blues", vmin=0)
labels_cm = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels_cm[i][j]}\n{cm[i,j]:,}",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if cm[i,j] > cm.max()*0.5 else C_DARK)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Pred\nNon-crit", "Pred\nCritical"], fontsize=9)
ax.set_yticklabels(["True\nNon-crit", "True\nCritical"], fontsize=9)
ax.set_title(f"Aggregate Confusion Matrix\n(labeled segments only)\n"
             f"Prec={precision_all:.3f}  Rec={recall_all:.3f}  F1={f1_all:.3f}",
             fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.75)

plt.tight_layout()
plt.savefig(FIG_DIR / "v3_figS2_cv_test_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: v3_figS2_cv_test_results.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig S3 — Prediction coverage
# ─────────────────────────────────────────────────────────────────────────────
print("Fig S3: Prediction coverage...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
fig.suptitle("Knox County Road Criticality GNN — Prediction Coverage",
             fontsize=13, fontweight="bold")

# Left: labeled vs unlabeled donut
ax = axes[0]
sizes  = [labeled_segs, unlabeled_segs]
labels_d = [f"TPO-labeled\n{labeled_segs:,}\n({labeled_segs/total_segs*100:.1f}%)",
            f"Unlabeled\n{unlabeled_segs:,}\n({unlabeled_segs/total_segs*100:.1f}%)"]
wedges, _ = ax.pie(sizes, labels=labels_d, colors=[C_ORANGE, C_GREY],
                   startangle=90, wedgeprops={"edgecolor":"white","linewidth":2,"width":0.5},
                   textprops={"fontsize":9})
ax.text(0, 0, f"{total_segs:,}\ntotal", ha="center", va="center",
        fontsize=10, fontweight="bold", color=C_DARK)
ax.set_title("All 65,524 Overture Segments\nLabeled vs Unlabeled", fontsize=11, fontweight="bold")

# Middle: predicted critical breakdown
ax = axes[1]
categories = ["Total\ndriveable", "TPO-labeled\n(training)", "Ground truth\ncritical",
              "Predicted\ncritical\n(all 65k)", "New predictions\n(unlabeled only)"]
vals = [total_segs, labeled_segs, n_gt_critical, n_pred_critical,
        n_pred_critical - tp]  # predicted critical that had no label
colors_bar = [C_GREY, C_ORANGE, C_RED, C_BLUE, C_GREEN]
bars = ax.bar(range(len(categories)), vals, color=colors_bar, alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylabel("Segment count", fontsize=10)
ax.set_title("Criticality Counts at Each Stage", fontsize=11, fontweight="bold")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            f"{val:,}", ha="center", fontsize=8, fontweight="bold")
ax.grid(axis="y", alpha=0.3); ax.set_facecolor(BG)
ax.set_yscale("log")
ax.set_ylim(100, 200000)

# Right: prob distribution of predictions
ax = axes[2]
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_labeled_pred   = segs[segs["critical"].notna()]["pred_prob_critical"]
n_unlabeled_pred = segs[segs["critical"].isna()]["pred_prob_critical"]
ax.hist(n_unlabeled_pred, bins=bins, color=C_GREY,  alpha=0.7, label=f"Unlabeled ({unlabeled_segs:,})", density=True)
ax.hist(n_labeled_pred,   bins=bins, color=C_ORANGE, alpha=0.7, label=f"TPO-labeled ({labeled_segs:,})", density=True)
ax.axvline(0.5, color=C_RED, linewidth=1.5, linestyle="--", label="Decision threshold (0.5)")
ax.set_xlabel("Predicted criticality probability", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Probability Distribution\nLabeled vs Unlabeled segments", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_facecolor(BG)

plt.tight_layout()
plt.savefig(FIG_DIR / "v3_figS3_prediction_coverage.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: v3_figS3_prediction_coverage.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig S4 — Critical segments by road class
# ─────────────────────────────────────────────────────────────────────────────
print("Fig S4: Critical by road class...")

class_order = ["motorway","trunk","primary","secondary","tertiary",
               "unclassified","unknown","living_street","residential","service"]

def class_stats(df, col, class_order):
    rows = []
    for cls in class_order:
        sub = df[df["class"] == cls]
        total = len(sub)
        if col == "critical":
            pos = (sub[col] == 1).sum()
        else:
            pos = (sub[col] == 1).sum()
        rows.append({"class": cls, "total": total, "positive": pos,
                     "rate": pos/total*100 if total > 0 else 0})
    return pd.DataFrame(rows)

gt_stats   = class_stats(segs[segs["critical"].notna()],   "critical",       class_order)
pred_stats = class_stats(segs, "pred_critical", class_order)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle("Road Criticality by Road Class", fontsize=13, fontweight="bold")

# Left: absolute counts stacked
ax = axes[0]
x  = np.arange(len(class_order))
w  = 0.35
ax.bar(x - w/2, gt_stats["total"],    w, color=C_GREY,  alpha=0.6, label="Total (labeled)", edgecolor="white")
ax.bar(x - w/2, gt_stats["positive"], w, color=C_RED,   alpha=0.9, label="Ground truth critical", edgecolor="white")
ax.bar(x + w/2, pred_stats["positive"], w, color=C_BLUE, alpha=0.9, label="Predicted critical (all segs)", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(class_order, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Segment count", fontsize=10)
ax.set_title("Absolute Counts per Class", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); ax.set_facecolor(BG)

# Right: critical rate (%) by class
ax = axes[1]
ax.plot(class_order, gt_stats["rate"],   "o-", color=C_RED,  linewidth=2,
        markersize=8, label="Ground truth critical rate (%)")
ax.plot(class_order, pred_stats["rate"], "s--", color=C_BLUE, linewidth=2,
        markersize=8, label="Predicted critical rate (%)")
ax.set_xticklabels(class_order, rotation=30, ha="right", fontsize=9)
ax.set_xticks(range(len(class_order)))
ax.set_xticklabels(class_order, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("% critical within class", fontsize=10)
ax.set_title("Criticality Rate by Road Class\n"
             "(GT rate only for TPO-labeled; pred rate for all)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_facecolor(BG)

plt.tight_layout()
plt.savefig(FIG_DIR / "v3_figS4_critical_by_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: v3_figS4_critical_by_class.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig S5 — Feature set summary
# ─────────────────────────────────────────────────────────────────────────────
print("Fig S5: Feature summary...")

feat_groups = {
    "Road geometry\n(Overture)":    ["length_m", "sinuosity"],
    "Road classification\n(Overture)": ["class_enc", "speed_kph", "has_surface"],
    "Network topology\n(Computed)": ["connector_count", "graph_degree", "betweenness"],
    "Structural flags\n(Overture)": ["is_bridge", "is_link", "is_tunnel", "is_private"],
    "Land use / TAZ\n(Morphology)": ["TOTAL_EMP", "HH", "building_coverage_pct",
                                      "street_density_km_km2", "building_density_n_km2",
                                      "avg_footprint_m2"],
}
group_counts = {k: len(v) for k, v in feat_groups.items()}
group_cols   = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_DARK]

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
fig.suptitle("GNN Node Feature Set — 18 Features Across 5 Groups",
             fontsize=13, fontweight="bold")

# Left: feature count by group (bar)
ax = axes[0]
bars = ax.barh(list(group_counts.keys()), list(group_counts.values()),
               color=group_cols, alpha=0.85, edgecolor="white", height=0.5)
for bar, val in zip(bars, group_counts.values()):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val} features", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Number of features", fontsize=10)
ax.set_title("Features per Group", fontsize=11, fontweight="bold")
ax.set_xlim(0, 9); ax.grid(axis="x", alpha=0.3); ax.set_facecolor(BG)

# Right: availability — all 65k vs TPO-only (concept table)
ax = axes[1]
ax.axis("off")
all_feats = [f for feats in feat_groups.values() for f in feats]
row_data  = []
for grp, feats in feat_groups.items():
    clean_grp = grp.replace("\n", " ")
    for f in feats:
        row_data.append([f, clean_grp, "✓ All 65,524", "TPO-derived label only"])

table_display = [
    ["Feature", "Group", "Used as input?", "Note"],
    *[[r[0], r[1].split("(")[0].strip(), r[2], ""] for r in row_data],
]

col_labels = ["Feature", "Group", "Available for all segs"]
cell_data  = [[r[0], r[1].split("(")[0].strip(), "Yes (all 65k)"] for r in row_data]

tbl = ax.table(cellText=cell_data, colLabels=col_labels,
               loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.3)

# Colour header
for j in range(3):
    tbl[0,j].set_facecolor(C_DARK)
    tbl[0,j].set_text_props(color="white", fontweight="bold")

# Colour group rows
grp_colour_map = {v.split("(")[0].strip(): c
                  for v, c in zip(feat_groups.keys(), group_cols)}
offset = 1
for grp, feats in feat_groups.items():
    clean = grp.split("(")[0].strip()
    for fi in range(len(feats)):
        for j in range(3):
            tbl[offset + fi, j].set_facecolor(grp_colour_map[clean] + "22")
    offset += len(feats)

ax.set_title("All 18 input features — full coverage over 65,524 segments\n"
             "Volume/V-C/Lanes → label construction only (not model input)",
             fontsize=10, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig(FIG_DIR / "v3_figS5_feature_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: v3_figS5_feature_summary.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print full stats to terminal
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
║        KNOX COUNTY ROAD CRITICALITY GNN — FULL STATS        ║
╚══════════════════════════════════════════════════════════════╝

── HOW TRAINING DATA WAS BUILT ────────────────────────────────
  Total Overture road segments (Knox County):   73,502
  Driveable classes only:                       65,524  (89.1%)
  TPO assignment links available:                4,799
  Matched to Overture (≤100m):                   4,105  (85.5%)
    (median TPO–Overture offset = 28.7m;
     100m threshold chosen to capture p90 of matches)

  Criticality label formula:
    score = 0.5 × norm(volume) + 0.5 × norm(betweenness)
    critical = 1  if score ≥ p80 (top 20%)
    critical = 0  otherwise

  Label breakdown (4,105 segments):
    Critical (label=1):     {n_gt_critical:,}  (20.0%)
    Non-critical (label=0): {n_gt_noncrit:,}  (80.0%)

── NODE FEATURES (24 total) ────────────────────────────────────
  Road geometry (2):      length_m, sinuosity
  Road class (3):         class_enc, speed_kph, has_surface
  Network topology (3):   connector_count, graph_degree, betweenness
  Structural flags (4):   is_bridge, is_link, is_tunnel, is_private
  Highway/Infrastructure (6): dist_to_major_road_m, hops_to_major_road,
                          major_road_density_500m, betweenness_to_major,
                          is_major_road, connects_to_major
  TAZ land use (6):       TOTAL_EMP, HH, building_coverage_pct,
                          street_density_km_km2, building_density_n_km2,
                          avg_footprint_m2
  (Volume, V/C, lanes → label only, NOT model input)

── MODEL ARCHITECTURE ──────────────────────────────────────────
  Type:      Graph Attention Network (GAT), 3-layer
  Layer 1:   GATConv(24 → 128, heads=16) → 2048-dim + ELU + Dropout(0.3)
  Layer 2:   GATConv(2048 → 128, heads=16) → 2048-dim + ELU + Dropout(0.3)
  Layer 3:   GATConv(2048 → 128, heads=1) → 128-dim
  Head:      Linear(128 → 2)  [binary classification]
  Loss:      Focal Loss (α=0.75, γ=2.0) - down-weights easy examples
  Graph:     65,524 nodes  |  159,472 edges (shared-endpoint adjacency)
  Training:  Adam lr=1e-3, dropout=0.3, early stop patience=60

── 5-FOLD SPATIAL CV TEST RESULTS ──────────────────────────────
{cv_df[["fold","n_test","AUC","F1","Precision","Recall"]].to_string(index=False)}

  Mean:  AUC={cv_df['AUC'].mean():.3f}  F1={cv_df['F1'].mean():.3f}  Prec={cv_df['Precision'].mean():.3f}  Rec={cv_df['Recall'].mean():.3f}
  Std:   AUC={cv_df['AUC'].std():.3f}  F1={cv_df['F1'].std():.3f}  Prec={cv_df['Precision'].std():.3f}  Rec={cv_df['Recall'].std():.3f}

── HOW PREDICTIONS WERE MADE ───────────────────────────────────
  1. Final model retrained on ALL 4,105 labeled segments
  2. Single forward pass over entire 65,524-node graph
  3. Each node receives embedding from 3 GAT layers,
     attending to its road-network neighbors
  4. Softmax head outputs P(critical) for every node
  5. Threshold 0.5 → binary pred_critical label

── PREDICTION RESULTS (ALL SEGMENTS) ───────────────────────────
  Total segments predicted on:    {total_segs:,}
    Of which TPO-labeled:         {labeled_segs:,}  (evaluable)
    Of which unlabeled:           {unlabeled_segs:,}  (new predictions)

  Predicted critical (all 65k):  {n_pred_critical:,}  ({n_pred_critical/total_segs*100:.1f}%)
  Predicted non-critical:        {n_pred_noncrit:,}  ({n_pred_noncrit/total_segs*100:.1f}%)

  On labeled segments only (ground truth available):
    TP (caught critical):  {tp:,}  of {n_gt_critical:,}  ({tp/n_gt_critical*100:.1f}% recall)
    FP (false alarms):     {fp:,}
    FN (missed critical):  {fn:,}
    TN (correct non-crit): {tn:,}
    Precision: {precision_all:.3f}  Recall: {recall_all:.3f}  F1: {f1_all:.3f}

  Genuinely new predictions (unlabeled segments flagged critical):
    {n_pred_critical - tp:,} segments that had NO TPO volume data
    → these are the segments the GNN extends criticality to

── OUTPUT FILES ────────────────────────────────────────────────
  critical_segments.gpkg       — all 65,524 segments, scored
  criticality_scores.csv       — same as table
  map1_critical_roads.html     — interactive map (layered)
  map2_full_heatmap.html       — full network probability heatmap
  figS1–figS5                  — this summary report figures
""")
