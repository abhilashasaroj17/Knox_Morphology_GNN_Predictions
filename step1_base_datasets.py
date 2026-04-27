"""
Knox County City2Graph — Step 1: Build Base Knox Datasets
==========================================================
Load, clean, reproject, validate, and export all Knox County data layers.

Run with:
    .venv\Scripts\python.exe step1_base_datasets.py
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# ── Data root paths ───────────────────────────────────────────────────────────
TAZ_DIR    = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph\data\KnoxTPO_Network_TAZShapefiles_OD")
ASSIGN_DIR = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph\data\Knox_Network_w_Attributes_Assignment")
OUT_DIR    = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph\outputs")

# ── Individual file paths ─────────────────────────────────────────────────────
ZONES_SHP   = TAZ_DIR    / "Knox_TAZ_shapefile.shp"
LINKS_SHP   = TAZ_DIR    / "Links_shapefile.shp"
OD_XLSX     = TAZ_DIR    / "2026_matrix to excel.xlsx"
ASSIGN_SHP  = ASSIGN_DIR / "Knox_Network w Attributes.shp"
ASSIGN_XLSX = ASSIGN_DIR / "2026_LinkFlows.xlsx"

TARGET_CRS  = "EPSG:26917"

# ── Join keys — UPDATE these after first run if column names differ ───────────
ZONE_ID_COL  = "TAZID"  # zone ID column in Knox_TAZ_shapefile.shp
ASSIGN_KEY_L = "ID"    # join key in assignment shapefile
ASSIGN_KEY_R = "ID1"   # join key in assignment volumes xlsx

# =============================================================================
# Section 1: Verify files
# =============================================================================
print("=" * 55)
print("  SECTION 1: File verification")
print("=" * 55)
all_ok = True
for f in [ZONES_SHP, LINKS_SHP, OD_XLSX, ASSIGN_SHP, ASSIGN_XLSX]:
    status = "OK" if f.exists() else "MISSING"
    if not f.exists():
        all_ok = False
    print(f"  {status}  {f.name}")
if not all_ok:
    raise FileNotFoundError("One or more input files are missing — check paths above.")
print("All files found.\n")

# =============================================================================
# Section 2: Load all data layers and inspect columns
# =============================================================================
print("=" * 55)
print("  SECTION 2: Loading data")
print("=" * 55)
zones_gdf  = gpd.read_file(ZONES_SHP)
links_gdf  = gpd.read_file(LINKS_SHP)
assign_gdf = gpd.read_file(ASSIGN_SHP)
od_raw     = pd.read_excel(OD_XLSX, index_col=0)
assign_vol = pd.read_excel(ASSIGN_XLSX)

print("ZONES columns :", zones_gdf.columns.tolist())
print("ZONES shape   :", zones_gdf.shape, "| CRS:", zones_gdf.crs)
print(zones_gdf.head(2), "\n")

print("LINKS columns :", links_gdf.columns.tolist())
print("LINKS shape   :", links_gdf.shape, "| CRS:", links_gdf.crs)
print(links_gdf.head(2), "\n")

print("ASSIGN NET columns:", assign_gdf.columns.tolist())
print("ASSIGN NET shape  :", assign_gdf.shape, "| CRS:", assign_gdf.crs)
print(assign_gdf.head(2), "\n")

print("OD MATRIX shape:", od_raw.shape)
print(od_raw.iloc[:3, :5], "\n")

print("ASSIGN VOL columns:", assign_vol.columns.tolist())
print("ASSIGN VOL shape  :", assign_vol.shape)
print(assign_vol.head(2), "\n")

# =============================================================================
# Section 3: Reproject to common CRS
# =============================================================================
print("=" * 55)
print("  SECTION 3: Reprojecting to", TARGET_CRS)
print("=" * 55)
zones_gdf  = zones_gdf.to_crs(TARGET_CRS)
links_gdf  = links_gdf.to_crs(TARGET_CRS)
assign_gdf = assign_gdf.to_crs(TARGET_CRS)
print("  zones :", zones_gdf.crs)
print("  links :", links_gdf.crs)
print("  assign:", assign_gdf.crs, "\n")

# =============================================================================
# Section 4: Assign zone_id and melt OD to long format
# =============================================================================
print("=" * 55)
print("  SECTION 4: zone_id + OD melt")
print("=" * 55)

# Auto-fallback if ZONE_ID_COL not found
if ZONE_ID_COL not in zones_gdf.columns:
    int_cols = [c for c in zones_gdf.columns if zones_gdf[c].dtype in [int, "int64", "int32"]]
    ZONE_ID_COL = int_cols[0]
    print(f"WARNING: ZONE_ID_COL not found — using fallback: {ZONE_ID_COL}")

zones_gdf = zones_gdf.rename(columns={ZONE_ID_COL: "zone_id"})
zones_gdf["zone_id"] = zones_gdf["zone_id"].astype(int)
print(f"zone_id dtype: {zones_gdf['zone_id'].dtype}")
print(f"Unique zones : {zones_gdf['zone_id'].nunique()}")
print(f"Duplicates   : {zones_gdf['zone_id'].duplicated().any()}\n")

od_raw = od_raw.apply(pd.to_numeric, errors="coerce")  # ensure numeric values
od_raw.index   = od_raw.index.astype(int)
od_raw.columns = od_raw.columns.astype(int)
od_stacked = od_raw.stack().reset_index()
od_stacked.columns = ["origin", "destination", "flow"]
od_long = od_stacked
od_long = od_long[od_long["flow"] > 0].reset_index(drop=True)
print(f"OD pairs (flow > 0): {len(od_long):,}")
print(f"Unique origins      : {od_long['origin'].nunique()}")
print(f"Unique destinations : {od_long['destination'].nunique()}")
print(f"Total flow          : {od_long['flow'].sum():,.0f}\n")

# =============================================================================
# Section 5: Join assignment volumes to network
# =============================================================================
print("=" * 55)
print("  SECTION 5: Join assignment volumes")
print("=" * 55)
if ASSIGN_KEY_L in assign_gdf.columns and ASSIGN_KEY_R in assign_vol.columns:
    assign_merged = assign_gdf.merge(
        assign_vol, left_on=ASSIGN_KEY_L, right_on=ASSIGN_KEY_R, how="left"
    )
    vol_cols = [c for c in assign_merged.columns if "FLOW" in c.upper() or "VOL" in c.upper()]
    n_joined = assign_merged[vol_cols[0]].notna().sum() if vol_cols else 0
    print(f"Links total         : {len(assign_gdf):,}")
    print(f"Links with volumes  : {n_joined:,}")
    print(f"Links missing vol.  : {len(assign_gdf) - n_joined:,}\n")
else:
    print(f"WARNING: Join keys '{ASSIGN_KEY_L}' / '{ASSIGN_KEY_R}' not found — skipping volume join.")
    print("  assign_gdf keys:", assign_gdf.columns.tolist())
    print("  assign_vol keys:", assign_vol.columns.tolist())
    assign_merged = assign_gdf.copy()

# =============================================================================
# Section 6: Step 1 Validation Table
# =============================================================================
print("=" * 55)
print("  SECTION 6: Step 1 Validation")
print("=" * 55)
valid_zone_ids    = set(zones_gdf["zone_id"].unique())
od_origins_valid  = od_long["origin"].isin(valid_zone_ids)
od_dest_valid     = od_long["destination"].isin(valid_zone_ids)
od_both_valid     = od_origins_valid & od_dest_valid
pct_od_valid      = od_both_valid.sum() / len(od_long) * 100

try:
    vol_col = [c for c in assign_merged.columns if "FLOW" in c.upper() or "VOL" in c.upper()]
    n_links_with_vol = assign_merged[vol_col[0]].notna().sum() if vol_col else 0
except Exception:
    n_links_with_vol = 0

val_df = pd.DataFrame({
    "Metric": [
        "Number of zones",
        "Number of OD pairs (flow > 0)",
        "Number of road links (TAZ folder)",
        "Number of assignment links",
        "Assignment links with volumes",
        "OD rows with valid zone IDs (%)"
    ],
    "Value": [
        zones_gdf["zone_id"].nunique(),
        len(od_long),
        len(links_gdf),
        len(assign_gdf),
        n_links_with_vol,
        f"{pct_od_valid:.1f}%"
    ]
})
print(val_df.to_string(index=False))

bad_origins = od_long[~od_origins_valid]["origin"].unique()
bad_dests   = od_long[~od_dest_valid]["destination"].unique()
if len(bad_origins) > 0:
    print(f"\nWARNING: OD origins not matching zone_id: {bad_origins[:10]}")
if len(bad_dests) > 0:
    print(f"WARNING: OD destinations not matching zone_id: {bad_dests[:10]}")
else:
    print("\nAll OD origins and destinations match zone IDs.")

# =============================================================================
# Export Step 1 outputs
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
od_long.to_csv(OUT_DIR / "od_long_2026.csv", index=False)
assign_merged.to_file(OUT_DIR / "assignment_with_volumes.gpkg", driver="GPKG")
zones_gdf.to_file(OUT_DIR / "zones_reprojected.gpkg", driver="GPKG")

print("\nStep 1 outputs saved to:", OUT_DIR)
print("  od_long_2026.csv")
print("  assignment_with_volumes.gpkg")
print("  zones_reprojected.gpkg")
print("\nStep 1 complete. Run step2_build_graphs.py next.")
