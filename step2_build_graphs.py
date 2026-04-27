"""
Knox County City2Graph � Step 2: Build OD Graph + Morphology Graph
===================================================================
Uses:
  - city2graph.data   : download roads + buildings from Overture Maps
  - city2graph.morphology : build morphological graph (buildings <-> streets)
  - networkx          : build OD graph from Step 1 outputs

Run AFTER step1_base_datasets.py:
    .venv\Scripts\python.exe step2_build_graphs.py
"""

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

from city2graph import data as c2g_data

OUT_DIR    = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph\outputs")
TARGET_CRS = "EPSG:26917"
PLACE      = "Knox County, Tennessee, USA"

# =============================================================================
# Load Step 1 outputs
# =============================================================================
print("Loading Step 1 outputs...")
zones_gdf     = gpd.read_file(OUT_DIR / "zones_reprojected.gpkg")
od_long       = pd.read_csv(OUT_DIR / "od_long_2026.csv")
assign_merged = gpd.read_file(OUT_DIR / "assignment_with_volumes.gpkg")
print(f"  Zones  : {len(zones_gdf):,}")
print(f"  OD rows: {len(od_long):,}")
print(f"  Links  : {len(assign_merged):,}\n")

# =============================================================================
# Section 7: Step 2a � Build OD Graph
# =============================================================================
print("=" * 55)
print("  SECTION 7: OD Graph")
print("=" * 55)

# Filter to internal-only OD pairs (both ends inside Knox TAZ zones)
valid_zone_ids = set(zones_gdf["zone_id"].unique())
od_internal = od_long[
    od_long["origin"].isin(valid_zone_ids) &
    od_long["destination"].isin(valid_zone_ids)
].copy()
print(f"Internal OD pairs (both ends in Knox): {len(od_internal):,}")
print(f"Dropped external pairs               : {len(od_long) - len(od_internal):,}\n")

# Zone-level OD summaries
productions = od_internal.groupby("origin")["flow"].sum().rename("production")
attractions = od_internal.groupby("destination")["flow"].sum().rename("attraction")
out_degree  = od_internal.groupby("origin")["destination"].count().rename("out_degree")
in_degree   = od_internal.groupby("destination")["origin"].count().rename("in_degree")

od_zone_stats = (
    zones_gdf[["zone_id"]].set_index("zone_id")
    .join(productions, how="left")
    .join(attractions, how="left")
    .join(out_degree, how="left")
    .join(in_degree, how="left")
    .fillna(0)
)
od_zone_stats["weighted_degree"] = od_zone_stats["out_degree"] + od_zone_stats["in_degree"]
od_zone_stats["net_flow"] = od_zone_stats["production"] - od_zone_stats["attraction"]

print("Top 10 zones by production:")
print(od_zone_stats.sort_values("production", ascending=False).head(10).to_string())

# Build NetworkX DiGraph
G_od = nx.DiGraph()
for _, row in od_internal.iterrows():
    G_od.add_edge(int(row["origin"]), int(row["destination"]), weight=row["flow"])
print(f"\nOD Graph - Nodes: {G_od.number_of_nodes():,}  |  Edges: {G_od.number_of_edges():,}")

od_internal.to_csv(OUT_DIR / "od_long_2026_internal.csv", index=False)
print("Saved: od_long_2026_internal.csv\n")

# =============================================================================
# Section 8: Step 2b � Fetch Overture Maps data (cached)
# =============================================================================
print("=" * 55)
print("  SECTION 8: Fetch Overture Maps data")
print("=" * 55)

print(f"Getting boundary for: {PLACE}")
boundary_gdf = c2g_data.get_boundaries(PLACE)
print(f"Boundary CRS: {boundary_gdf.crs}")

print("Downloading Overture Maps data (segment, building, connector)...")
print("This may take several minutes. Data is cached to outputs/overture_cache/")
overture_data = c2g_data.load_overture_data(
    area=boundary_gdf,
    types=["segment", "building", "connector"],
    output_dir=str(OUT_DIR / "overture_cache"),
    prefix="knox_",
    save_to_file=True,
    return_data=True,
)

segments_raw  = overture_data.get("segment",   gpd.GeoDataFrame())
buildings_raw = overture_data.get("building",  gpd.GeoDataFrame())
connectors    = overture_data.get("connector", gpd.GeoDataFrame())

print(f"Overture segments  : {len(segments_raw):,}")
print(f"Overture buildings : {len(buildings_raw):,}")
print(f"Overture connectors: {len(connectors):,}\n")

# =============================================================================
# Section 9: Process Overture segments
# =============================================================================
print("=" * 55)
print("  SECTION 9: Process Overture road segments")
print("=" * 55)
segments_proc = c2g_data.process_overture_segments(
    segments_raw,
    get_barriers=True,
    connectors_gdf=connectors if len(connectors) > 0 else None,
)
print(f"Processed segments : {len(segments_proc):,}")
print(f"Segment columns    : {segments_proc.columns.tolist()}\n")

segments_proj  = segments_proc.to_crs(TARGET_CRS)
buildings_proj = buildings_raw.to_crs(TARGET_CRS)
buildings_proj = buildings_proj[
    buildings_proj.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()
print(f"Buildings (polygon): {len(buildings_proj):,}")

# =============================================================================
# Section 10: NOTE on Morphological Graph
# =============================================================================
# city2graph's morphological_graph() builds tessellations + building-street
# adjacency, but requires union_all of all segments — infeasible county-wide
# (200k buildings, 135k segments -> OOM). For research purposes this should be
# run per-zone or per-neighbourhood. For TAZ-level feature extraction we go
# directly to spatial aggregation in Section 11, which is what we need.

print("=" * 55)
print("  NOTE: Skipping county-wide morphological graph")
print("  (too large; run per-zone for research use)")
print("  Proceeding to TAZ-level spatial aggregation...")
print("=" * 55)

# =============================================================================
# Section 11: Aggregate morphology to TAZ zones
# =============================================================================
print("\n" + "=" * 55)
print("  SECTION 11: Aggregate morphology to TAZ zones")
print("=" * 55)

segments_proj["seg_length_m"] = segments_proj.geometry.length
edges_in_zones = gpd.sjoin(
    segments_proj[["geometry", "seg_length_m"]].reset_index(drop=True),
    zones_gdf[["zone_id", "geometry"]],
    how="left", predicate="intersects"
)
morph_roads = edges_in_zones.groupby("zone_id").agg(
    n_segments=("seg_length_m", "count"),
    total_length_m=("seg_length_m", "sum"),
    avg_seg_length_m=("seg_length_m", "mean"),
).reset_index()

buildings_proj["footprint_m2"] = buildings_proj.geometry.area
bldg_in_zones = gpd.sjoin(
    buildings_proj[["geometry", "footprint_m2"]].reset_index(drop=True),
    zones_gdf[["zone_id", "geometry"]],
    how="left", predicate="intersects"
)
morph_buildings = bldg_in_zones.groupby("zone_id").agg(
    n_buildings=("footprint_m2", "count"),
    total_footprint_m2=("footprint_m2", "sum"),
    avg_footprint_m2=("footprint_m2", "mean"),
).reset_index()

zones_gdf["area_km2"] = zones_gdf.geometry.area / 1e6
morph = zones_gdf[["zone_id", "area_km2"]].merge(morph_roads, on="zone_id", how="left")
morph = morph.merge(morph_buildings, on="zone_id", how="left").fillna(0)
morph["street_density_km_km2"]  = morph["total_length_m"] / 1000 / morph["area_km2"]
morph["building_density_n_km2"] = morph["n_buildings"] / morph["area_km2"]
morph["building_coverage_pct"]  = morph["total_footprint_m2"] / (morph["area_km2"] * 1e6) * 100

print("Top 10 zones by street density:")
print(morph.sort_values("street_density_km_km2", ascending=False).head(10).to_string(index=False))

# =============================================================================
# Section 12: Export
# =============================================================================
print("\n" + "=" * 55)
print("  SECTION 12: Export")
print("=" * 55)

zones_out = zones_gdf.merge(morph.drop(columns=["area_km2"]), on="zone_id", how="left")
zones_out = zones_out.merge(od_zone_stats.reset_index(), on="zone_id", how="left")
zones_out.to_file(OUT_DIR / "zones_clean.gpkg", driver="GPKG")

morph.to_csv(OUT_DIR / "morphology_zone_features.csv", index=False)
od_zone_stats.reset_index().to_csv(OUT_DIR / "od_zone_stats.csv", index=False)

print(f"  zones_clean.gpkg             : {len(zones_out):,} zones")
print(f"  morphology_zone_features.csv : {len(morph):,} zones")
print(f"  od_zone_stats.csv            : {len(od_zone_stats):,} zones")
print(f"\nFinal columns in zones_clean.gpkg:\n  {zones_out.columns.tolist()}")
print("\nStep 2 complete. Next: step3_regression.py")
