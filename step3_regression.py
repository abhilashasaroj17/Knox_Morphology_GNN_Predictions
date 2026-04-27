"""
Knox County City2Graph - Step 3: Regression Analysis
=====================================================
Models trip production and attraction as a function of:
  - Land use controls: TOTAL_EMP, TOTPOP, HH, area_km2
  - Morphology features: building_coverage_pct, avg_seg_length_m,
                         street_density_km_km2, building_density_n_km2

Models:
  1. OLS (baseline)
  2. Poisson GLM (count-appropriate)
  3. Spatial 5-fold cross-validation of both models

Outputs (outputs/regression/):
  - reg_ols_production.txt / reg_ols_attraction.txt
  - reg_poisson_production.txt / reg_poisson_attraction.txt
  - figR1_coefficients.png        — standardized coefficient comparison
  - figR2_residual_maps.png       — spatial residual maps (OLS)
  - figR3_partial_plots.png       — partial regression plots
  - figR4_cv_results.png          — k-fold CV performance (OLS vs Poisson)
  - figR5_predicted_vs_actual.png — scatter of predictions
  - cv_results.csv                — per-fold metrics

Run with:
    $env:PYTHONUTF8="1"; .venv\\Scripts\\Activate.ps1; python step3_regression.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# Paths
# =============================================================================
ROOT   = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph")
OUT    = ROOT / "outputs"
REG_DIR = OUT / "regression"
FIG_DIR = OUT / "figures"
REG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Section 1: Load & merge data
# =============================================================================
print("=" * 60)
print("  SECTION 1: Load data")
print("=" * 60)

zones  = gpd.read_file(OUT / "zones_clean.gpkg")
morph  = pd.read_csv(OUT / "morphology_zone_features.csv")
od     = pd.read_csv(OUT / "od_zone_stats.csv")

df = zones[["zone_id", "TOTPOP", "HH", "TOTAL_EMP", "geometry"]].copy()
df = df.merge(morph, on="zone_id", how="left")
df = df.merge(od[["zone_id", "production", "attraction"]], on="zone_id", how="left")

# Centroid coordinates for spatial fold assignment
df["cx"] = df.geometry.centroid.x
df["cy"] = df.geometry.centroid.y

# Drop zones with zero production/attraction (external/dummy zones)
df = df[(df["production"] > 0) & (df["attraction"] > 0)].copy()
df = df.fillna(0)
print(f"  Zones for regression: {len(df)}")
print(f"  Production range: {df['production'].min():.0f} – {df['production'].max():.0f}")
print(f"  Attraction range: {df['attraction'].min():.0f} – {df['attraction'].max():.0f}")

# =============================================================================
# Section 2: Feature sets
# =============================================================================
CONTROLS  = ["TOTAL_EMP", "TOTPOP", "HH", "area_km2"]
MORPHOLOGY = ["building_coverage_pct", "avg_seg_length_m",
              "street_density_km_km2", "building_density_n_km2"]
ALL_FEATS = CONTROLS + MORPHOLOGY
TARGETS   = ["production", "attraction"]

# Log-transform skewed variables for OLS
for col in ["TOTAL_EMP", "TOTPOP", "HH", "production", "attraction"]:
    df[f"log_{col}"] = np.log1p(df[col])

# =============================================================================
# Section 3: VIF check (multicollinearity)
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 3: VIF check")
print("=" * 60)

X_vif = df[ALL_FEATS].copy().fillna(0)
X_vif = sm.add_constant(X_vif)
vif_data = pd.DataFrame({
    "feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print(vif_data.to_string(index=False))
vif_data.to_csv(REG_DIR / "vif_table.csv", index=False)

# =============================================================================
# Section 4: OLS Regression (log-log for production & attraction)
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 4: OLS Regression")
print("=" * 60)

log_controls   = [f"log_{c}" if c in ["TOTAL_EMP","TOTPOP","HH"] else c for c in CONTROLS]
ols_feats      = log_controls + MORPHOLOGY

ols_results = {}
for target in TARGETS:
    y = df[f"log_{target}"]
    X = sm.add_constant(df[ols_feats].fillna(0))
    model = sm.OLS(y, X).fit()
    ols_results[target] = model
    summary_txt = model.summary().as_text()
    (REG_DIR / f"reg_ols_{target}.txt").write_text(summary_txt)
    print(f"\n  OLS {target}: R²={model.rsquared:.3f}, adj-R²={model.rsquared_adj:.3f}, "
          f"AIC={model.aic:.1f}")
    sig = model.pvalues[model.pvalues < 0.05].index.tolist()
    print(f"    Significant (p<0.05): {sig}")

# =============================================================================
# Section 5: Poisson GLM
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 5: Poisson GLM")
print("=" * 60)

poisson_results = {}
for target in TARGETS:
    y = df[target].round().astype(int)   # Poisson needs integer counts
    X = sm.add_constant(df[ALL_FEATS].fillna(0))
    model = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.Log())).fit()
    poisson_results[target] = model
    summary_txt = model.summary().as_text()
    (REG_DIR / f"reg_poisson_{target}.txt").write_text(summary_txt)
    # Pseudo R² (McFadden)
    pseudo_r2 = 1 - model.llf / model.llnull
    print(f"\n  Poisson {target}: pseudo-R²={pseudo_r2:.3f}, AIC={model.aic:.1f}, "
          f"deviance={model.deviance:.1f}")
    sig = model.pvalues[model.pvalues < 0.05].index.tolist()
    print(f"    Significant (p<0.05): {sig}")

# =============================================================================
# Section 6: Spatial 5-fold Cross-Validation
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 6: Spatial 5-fold Cross-Validation")
print("=" * 60)

K = 5
# Assign spatial folds by clustering zone centroids
km = KMeans(n_clusters=K, random_state=42, n_init=10)
df["spatial_fold"] = km.fit_predict(df[["cx", "cy"]].values)
print(f"  Fold sizes: {df['spatial_fold'].value_counts().sort_index().to_dict()}")

cv_rows = []
df_cv = df.reset_index(drop=True)
scaler = StandardScaler()

for fold in range(K):
    test_mask  = df_cv["spatial_fold"] == fold
    train_mask = ~test_mask
    train_df   = df_cv[train_mask].copy()
    test_df    = df_cv[test_mask].copy()

    for target in TARGETS:
        # --- OLS ---
        y_train = train_df[f"log_{target}"]
        X_train = sm.add_constant(train_df[ols_feats].fillna(0), has_constant="add")
        X_test  = sm.add_constant(test_df[ols_feats].fillna(0),  has_constant="add")
        # Align columns
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        ols_m   = sm.OLS(y_train, X_train).fit()
        y_pred_log = ols_m.predict(X_test)
        y_pred_ols = np.expm1(y_pred_log)
        y_true     = test_df[target].values

        mae  = mean_absolute_error(y_true, y_pred_ols)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_ols))
        r2   = r2_score(y_true, y_pred_ols)
        cv_rows.append({"fold": fold, "target": target, "model": "OLS",
                        "MAE": mae, "RMSE": rmse, "R2": r2,
                        "n_test": test_mask.sum()})

        # --- Poisson ---
        y_train_p = train_df[target].round().astype(int)
        X_train_p = sm.add_constant(train_df[ALL_FEATS].fillna(0), has_constant="add")
        X_test_p  = sm.add_constant(test_df[ALL_FEATS].fillna(0),  has_constant="add")
        X_test_p  = X_test_p.reindex(columns=X_train_p.columns, fill_value=0)
        try:
            pois_m     = sm.GLM(y_train_p, X_train_p,
                                family=sm.families.Poisson()).fit(maxiter=100)
            y_pred_pois = pois_m.predict(X_test_p)
            mae_p  = mean_absolute_error(y_true, y_pred_pois)
            rmse_p = np.sqrt(mean_squared_error(y_true, y_pred_pois))
            r2_p   = r2_score(y_true, y_pred_pois)
        except Exception as e:
            print(f"    Poisson fold {fold} {target} failed: {e}")
            mae_p, rmse_p, r2_p = np.nan, np.nan, np.nan
        cv_rows.append({"fold": fold, "target": target, "model": "Poisson",
                        "MAE": mae_p, "RMSE": rmse_p, "R2": r2_p,
                        "n_test": test_mask.sum()})

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(REG_DIR / "cv_results.csv", index=False)

summary_cv = cv_df.groupby(["model", "target"])[["MAE", "RMSE", "R2"]].agg(
    ["mean", "std"]).round(2)
print("\n  Cross-validation summary:")
print(summary_cv.to_string())

# =============================================================================
# Section 7: Figure R1 — Standardized Coefficients
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 7: Figures")
print("=" * 60)
print("  Fig R1: Standardized coefficients...")

scaler_f = StandardScaler()
X_std = pd.DataFrame(
    scaler_f.fit_transform(df[ols_feats].fillna(0)),
    columns=ols_feats
)
std_coefs = {}
for target in TARGETS:
    y = df[f"log_{target}"]
    X = sm.add_constant(X_std)
    m = sm.OLS(y, X).fit()
    coef = m.params.drop("const")
    ci   = m.conf_int().drop("const")
    std_coefs[target] = {"coef": coef, "ci": ci, "pval": m.pvalues.drop("const")}

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
feat_labels = {
    "log_TOTAL_EMP":   "log(Employment)",
    "log_TOTPOP":      "log(Population)",
    "log_HH":          "log(Households)",
    "area_km2":        "Zone area (km²)",
    "building_coverage_pct":  "Building coverage %",
    "avg_seg_length_m":       "Avg segment length (m)",
    "street_density_km_km2":  "Street density (km/km²)",
    "building_density_n_km2": "Building density (n/km²)",
}
separator_after = "area_km2"

for ax, target in zip(axes, TARGETS):
    coef = std_coefs[target]["coef"]
    ci   = std_coefs[target]["ci"]
    pval = std_coefs[target]["pval"]
    labels = [feat_labels.get(f, f) for f in coef.index]
    y_pos  = np.arange(len(labels))
    colors = ["#E74C3C" if c > 0 else "#3498DB" for c in coef.values]
    sig    = pval.values < 0.05

    for yi, (cv, col, sv) in enumerate(zip(coef.values, colors, sig)):
        ax.barh(yi, cv, color=col, alpha=0.9 if sv else 0.4,
                edgecolor="white", height=0.6)
    ax.errorbar(coef.values, y_pos,
                xerr=[coef.values - ci[0].values, ci[1].values - coef.values],
                fmt="none", color="black", linewidth=1, capsize=3)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")

    # Separator line between controls and morphology
    sep_idx = list(coef.index).index(separator_after) + 0.5
    ax.axhline(sep_idx, color="gray", linewidth=0.8, linestyle=":")
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -0.5,
            sep_idx + 0.1, "↑ land use   ↓ morphology",
            fontsize=7, color="gray", va="bottom")

    # Significance stars
    for i, (c, s) in enumerate(zip(coef.values, sig)):
        if s:
            ax.text(c + 0.01 if c > 0 else c - 0.01, i, "*",
                    ha="left" if c > 0 else "right", va="center",
                    fontsize=11, color="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Standardized coefficient (β)", fontsize=10)
    ax.set_title(f"Trip {target.capitalize()}\n(OLS on log-transformed target)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("Knox County: Standardized OLS Coefficients\n"
             "Red = positive effect | Blue = negative | * = p<0.05 | Faded = not significant",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figR1_coefficients.png", dpi=160, bbox_inches="tight")
plt.close()
print("    Saved: figR1_coefficients.png")

# =============================================================================
# Fig R2: Spatial Residual Maps
# =============================================================================
print("  Fig R2: Spatial residual maps...")

df_geo = df.copy()
for target in TARGETS:
    model = ols_results[target]
    X_full = sm.add_constant(df[ols_feats].fillna(0))
    df_geo[f"resid_{target}"] = model.resid.values
    df_geo[f"pred_{target}"]  = np.expm1(model.fittedvalues.values)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, target in zip(axes, TARGETS):
    resid = df_geo[f"resid_{target}"]
    vmax  = np.percentile(np.abs(resid), 95)
    norm  = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    gdf_plot = gpd.GeoDataFrame(df_geo[[f"resid_{target}", "geometry"]], geometry="geometry")
    gdf_plot.plot(column=f"resid_{target}", ax=ax, cmap="RdBu_r", norm=norm,
                  legend=True,
                  legend_kwds={"label": "Residual (log scale)", "shrink": 0.6},
                  edgecolor="white", linewidth=0.2, missing_kwds={"color": "lightgrey"})
    ax.set_title(f"OLS Residuals — Trip {target.capitalize()}\n"
                 "Red = over-predicted  |  Blue = under-predicted",
                 fontsize=11, fontweight="bold")
    ax.set_axis_off()

fig.suptitle("Knox County OLS Regression: Spatial Residuals\n"
             "Clusters indicate spatial autocorrelation → motivation for GNN",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figR2_residual_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figR2_residual_maps.png")

# =============================================================================
# Fig R3: Partial Regression Plots (morphology vars only)
# =============================================================================
print("  Fig R3: Partial regression plots...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for row_idx, target in enumerate(TARGETS):
    model = ols_results[target]
    for col_idx, feat in enumerate(MORPHOLOGY):
        ax = axes[row_idx][col_idx]
        # Partial regression plot: residuals of y on X-feat vs residuals of feat on X-rest
        other_feats = [f for f in ols_feats if f != feat]
        X_other = sm.add_constant(df[other_feats].fillna(0))
        y_log   = df[f"log_{target}"]

        resid_y    = sm.OLS(y_log,      X_other).fit().resid
        resid_feat = sm.OLS(df[feat].fillna(0), X_other).fit().resid

        ax.scatter(resid_feat, resid_y, s=8, alpha=0.4, color="#2C3E50", edgecolors="none")
        # Fit line
        fit = np.polyfit(resid_feat, resid_y, 1)
        xr  = np.linspace(resid_feat.min(), resid_feat.max(), 100)
        ax.plot(xr, np.polyval(fit, xr), "r-", linewidth=1.5)

        r_val = np.corrcoef(resid_feat, resid_y)[0, 1]
        ax.set_title(f"{feat_labels.get(feat, feat)}\nr_partial = {r_val:.3f}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Residual (feature)", fontsize=7)
        ax.set_ylabel("Residual (log y)" if col_idx == 0 else "", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    axes[row_idx][0].set_ylabel(f"log({target})\nresidual", fontsize=9)

fig.suptitle("Partial Regression Plots — Morphology Features\n"
             "(controlling for employment, population, households, area)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figR3_partial_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figR3_partial_plots.png")

# =============================================================================
# Fig R4: Cross-Validation Results
# =============================================================================
print("  Fig R4: CV results...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = ["MAE", "RMSE", "R2"]
metric_labels = {"MAE": "MAE (trips)", "RMSE": "RMSE (trips)", "R2": "R²"}
colors_model = {"OLS": "#E74C3C", "Poisson": "#2ECC71"}

for ax, metric in zip(axes, metrics):
    for model_name in ["OLS", "Poisson"]:
        for t_idx, target in enumerate(TARGETS):
            vals = cv_df[(cv_df["model"] == model_name) & (cv_df["target"] == target)][metric].dropna()
            x_pos = t_idx * 3 + (0 if model_name == "OLS" else 1)
            ax.bar(x_pos, vals.mean(), width=0.8,
                   color=colors_model[model_name], alpha=0.7 if target == "production" else 0.45,
                   label=f"{model_name} {target}" if ax == axes[0] else None,
                   edgecolor="white")
            ax.errorbar(x_pos, vals.mean(), yerr=vals.std(),
                        fmt="none", color="black", capsize=4, linewidth=1.2)

    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["Production", "Attraction"], fontsize=10)
    ax.set_ylabel(metric_labels[metric], fontsize=10)
    ax.set_title(f"5-Fold CV: {metric_labels[metric]}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

handles = [plt.Rectangle((0,0),1,1, color=colors_model[m], alpha=a)
           for m, a in [("OLS",0.7),("OLS",0.45),("Poisson",0.7),("Poisson",0.45)]]
labels  = ["OLS Production","OLS Attraction","Poisson Production","Poisson Attraction"]
axes[2].legend(handles, labels, fontsize=8, loc="lower right")

fig.suptitle("Knox County — Spatial 5-Fold Cross-Validation\n"
             "OLS vs Poisson  |  Error bars = ±1 std across folds",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figR4_cv_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figR4_cv_results.png")

# =============================================================================
# Fig R5: Predicted vs Actual
# =============================================================================
print("  Fig R5: Predicted vs actual...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
model_map = {"OLS": ols_results, "Poisson": poisson_results}

for row_idx, target in enumerate(TARGETS):
    for col_idx, (model_name, mdict) in enumerate(model_map.items()):
        ax = axes[row_idx][col_idx]
        model = mdict[target]
        if model_name == "OLS":
            y_pred = np.expm1(model.fittedvalues.values)
        else:
            y_pred = model.fittedvalues.values
        y_true = df[target].values

        ax.scatter(y_true, y_pred, s=12, alpha=0.45, color="#2C3E50", edgecolors="none")
        lim = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect fit")
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax.set_title(f"{model_name} — {target.capitalize()}\nR²={r2:.3f}  RMSE={rmse:.0f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual trips", fontsize=10)
        ax.set_ylabel("Predicted trips", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        # Annotate worst residuals
        resids = np.abs(y_true - y_pred)
        top5   = np.argsort(resids)[-5:]
        for i in top5:
            ax.annotate(str(df["zone_id"].iloc[i]),
                        (y_true[i], y_pred[i]), fontsize=6, alpha=0.7,
                        xytext=(4, 4), textcoords="offset points")

fig.suptitle("Knox County: Predicted vs Actual Trip Counts\n"
             "Labelled points = largest residuals (zone IDs)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "figR5_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: figR5_predicted_vs_actual.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 3 COMPLETE")
print("=" * 60)
print(f"\nOLS results:")
for t in TARGETS:
    m = ols_results[t]
    print(f"  {t:12s}  R²={m.rsquared:.3f}  adj-R²={m.rsquared_adj:.3f}")
print(f"\nPoisson results:")
for t in TARGETS:
    m = poisson_results[t]
    pr2 = 1 - m.llf / m.llnull
    print(f"  {t:12s}  pseudo-R²={pr2:.3f}  AIC={m.aic:.1f}")
print(f"\nCV summary (mean across 5 spatial folds):")
print(cv_df.groupby(["model","target"])[ ["MAE","RMSE","R2"]].mean().round(2).to_string())
print(f"\nAll regression outputs saved to: {REG_DIR}")
print(f"All figures saved to:            {FIG_DIR}")
print("\nNext: step4_hetero_graph.py")
