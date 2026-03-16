#!/usr/bin/env python3
"""
make_arcgis_geojson.py
======================
Produces two ArcGIS-ready GeoJSON files from the distribution network
shapefiles and load-profile CSVs.

OUTPUT
------
output/arcgis_substations_redday.geojson
    21 Point features — one per substation.
    Attributes:
        substation_id          — region folder name (e.g. "121579")
        diff_h01 … diff_h24   — Tempo-shifted minus Control load (kWh)
                                  aggregated across all HID meters that
                                  belong to this substation, for each
                                  hour of the Red Day (2014-07-03).

output/arcgis_edges_redday.geojson
    ~58,000 LineString features — every distribution-network edge.
    Attributes:
        substation_id          — which substation this edge belongs to
        diff_h01 … diff_h24   — same values as the parent substation
                                  (denormalized so ArcGIS can colour
                                  edges directly without a join)

CRS: EPSG:4326 (WGS84) — required for GeoJSON.

ARCGIS USAGE
------------
  • Add both layers via Add Data.
  • Symbolize on any diff_hXX field with a diverging Blue–White–Red ramp:
        Blue  = tempo-shifting reduced load
        Red   = tempo-shifting increased load
  • Step through hours manually, or use the Range Slider
    (Analysis → Range → Field: diff_h01…diff_h24).
  • To use the Time Slider instead, pivot to long format in ArcGIS:
        Data Engineering → Transpose Fields → diff_h01…diff_h24
        then enable Time on the resulting timestamp column.
"""

import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = '/home/user/capstone_visuals'
ZIP_PATH    = os.path.join(BASE_DIR, 'dist_net.zip')
CONTROL_CSV = os.path.join(BASE_DIR, 'data', 'control_profile.csv')
TEMPO_CSV   = os.path.join(BASE_DIR, 'data', 'tempo_shifted_profile.csv')
EXTRACT_DIR = '/tmp/dist_net'
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RED_DAY   = '2014-07-03'
HOUR_COLS = [str(i) for i in range(1, 25)]
DIFF_COLS = [f'diff_h{h:02d}' for h in range(1, 25)]   # diff_h01 … diff_h24


# ─── Step 1: Extract zip ──────────────────────────────────────────────────────
def extract_zip():
    flag = os.path.join(EXTRACT_DIR, '.done')
    if not os.path.exists(flag):
        print('Extracting dist_net.zip …')
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(EXTRACT_DIR)
        open(flag, 'w').close()
    else:
        print('Network archive already extracted.')


# ─── Step 2: Load shapefiles ──────────────────────────────────────────────────
def load_network():
    content = os.path.join(EXTRACT_DIR, 'content', 'output')
    all_nodes, all_edges = [], []
    for folder in sorted(os.listdir(content)):
        fp = os.path.join(content, folder)
        if not os.path.isdir(fp):
            continue
        ns = os.path.join(fp, f'{folder}-nodelist-HID.shp')
        es = os.path.join(fp, f'{folder}-edgelist.shp')
        if os.path.exists(ns):
            g = gpd.read_file(ns); g['region'] = folder; all_nodes.append(g)
        if os.path.exists(es):
            g = gpd.read_file(es); g['region'] = folder; all_edges.append(g)

    nodes = gpd.GeoDataFrame(pd.concat(all_nodes, ignore_index=True),
                             geometry='geometry', crs=all_nodes[0].crs)
    edges = gpd.GeoDataFrame(pd.concat(all_edges, ignore_index=True),
                             geometry='geometry', crs=all_edges[0].crs)

    # Ensure WGS84
    nodes = nodes.to_crs('EPSG:4326')
    edges = edges.to_crs('EPSG:4326')

    print(f'  {len(nodes):,} nodes  |  {len(edges):,} edges  |  '
          f'{nodes["region"].nunique()} substations')
    return nodes, edges


def normalize_hid(x):
    try:    return str(int(float(x)))
    except: return None


# ─── Step 3: Compute per-substation hourly diff on Red Day ───────────────────
def compute_diff_by_substation(nodes):
    """
    Returns a DataFrame:
        substation_id | diff_h01 | diff_h02 | … | diff_h24
    one row per substation (21 rows).

    diff_hXX = Σ tempo_load(hid, h) − Σ ctrl_load(hid, h)
               summed over all HID meters belonging to that substation.
    """
    print('\nLoading CSVs …')
    ctrl_df  = pd.read_csv(CONTROL_CSV)
    tempo_df = pd.read_csv(TEMPO_CSV)
    ctrl_df ['hid_key'] = ctrl_df ['hid'].apply(normalize_hid)
    tempo_df['hid_key'] = tempo_df['hid'].apply(normalize_hid)

    # Filter to Red Day only
    ctrl_red  = ctrl_df [ctrl_df ['date'] == RED_DAY].copy()
    tempo_red = tempo_df[tempo_df['date'] == RED_DAY].copy()
    print(f'  Control  HIDs on Red Day: {ctrl_red ["hid_key"].nunique():,}')
    print(f'  Tempo    HIDs on Red Day: {tempo_red["hid_key"].nunique():,}')

    # Household nodes → region mapping
    hh = nodes[nodes['label'] == 'H'][['hid', 'region']].copy()
    hh['hid_key'] = hh['hid'].apply(normalize_hid)
    hh = hh.dropna(subset=['hid_key'])

    def region_totals(csv_red):
        """Merge CSV → household nodes; sum load by region and hour."""
        merged = hh.merge(csv_red[['hid_key'] + HOUR_COLS],
                          on='hid_key', how='inner')
        totals = merged.groupby('region')[HOUR_COLS].sum()
        return totals   # DataFrame: region × hours

    ctrl_totals  = region_totals(ctrl_red)
    tempo_totals = region_totals(tempo_red)

    # Align on the same region index; fill missing with 0
    all_regions = sorted(nodes['region'].unique())
    ctrl_totals  = ctrl_totals .reindex(all_regions, fill_value=0)
    tempo_totals = tempo_totals.reindex(all_regions, fill_value=0)

    diff = tempo_totals - ctrl_totals   # element-wise difference
    diff.columns = DIFF_COLS            # rename 1…24 → diff_h01…diff_h24
    diff.index.name = 'substation_id'
    diff = diff.reset_index()

    print(f'\n  Diff range: '
          f'{diff[DIFF_COLS].values.min():.2f} … '
          f'{diff[DIFF_COLS].values.max():.2f} kWh')
    print(f'  Non-zero substations per hour (mean): '
          f'{(diff[DIFF_COLS] != 0).sum(axis=0).mean():.1f} / {len(diff)}')
    return diff


# ─── Step 4: Build substation GeoJSON (21 Point features) ────────────────────
def build_substation_geojson(nodes, diff_df):
    """One Point feature per substation 'S' node with all diff columns."""
    subs = nodes[nodes['label'] == 'S'].copy()
    subs = subs.rename(columns={'region': 'substation_id'})
    subs = subs[['substation_id', 'geometry']].reset_index(drop=True)

    # Merge diff values
    subs = subs.merge(diff_df, on='substation_id', how='left')

    # Fill any substations with no matched HIDs with 0
    subs[DIFF_COLS] = subs[DIFF_COLS].fillna(0.0)

    # Round to 4 decimal places (sufficient for kWh)
    subs[DIFF_COLS] = subs[DIFF_COLS].round(4)

    print(f'\n  Substation features:  {len(subs)} '
          f'(expected 21, got {len(subs)})')
    return gpd.GeoDataFrame(subs, geometry='geometry', crs='EPSG:4326')


# ─── Step 5: Build edge GeoJSON (~58k LineString features) ───────────────────
def build_edge_geojson(edges, diff_df):
    """
    All distribution edges, each tagged with its parent substation_id and
    the same diff_h01…diff_h24 values as that substation (denormalized).

    MultiLineString geometries are exploded to individual LineStrings so
    every feature has a simple geometry — required for GeoJSON.
    """
    edge_gdf = edges.rename(columns={'region': 'substation_id'}).copy()
    edge_gdf = edge_gdf[['substation_id', 'geometry']].reset_index(drop=True)

    # Explode MultiLineString → LineString
    edge_gdf = edge_gdf.explode(index_parts=False).reset_index(drop=True)

    # Merge diff values from parent substation (denormalize)
    edge_gdf = edge_gdf.merge(diff_df, on='substation_id', how='left')
    edge_gdf[DIFF_COLS] = edge_gdf[DIFF_COLS].fillna(0.0).round(4)

    # Drop any degenerate geometries (null or empty)
    edge_gdf = edge_gdf[~edge_gdf.geometry.is_empty & edge_gdf.geometry.notna()]
    edge_gdf = edge_gdf.reset_index(drop=True)

    print(f'  Edge features:  {len(edge_gdf):,}')
    return gpd.GeoDataFrame(edge_gdf, geometry='geometry', crs='EPSG:4326')


# ─── Step 6: Write GeoJSON files ─────────────────────────────────────────────
def write_geojson(gdf, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    gdf.to_file(path, driver='GeoJSON')
    size_kb = os.path.getsize(path) // 1024
    print(f'  Wrote → {path}  ({len(gdf):,} features, {size_kb:,} KB)')
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=== ArcGIS GeoJSON Export  —  Red Day Difference ===')
    print(f'    Red Day: {RED_DAY}   |   Diff = Tempo-shifted − Control\n')

    extract_zip()

    print('\nLoading shapefiles …')
    nodes, edges = load_network()

    diff_df = compute_diff_by_substation(nodes)

    print('\nBuilding GeoJSON layers …')
    sub_gdf  = build_substation_geojson(nodes, diff_df)
    edge_gdf = build_edge_geojson(edges, diff_df)

    print('\nWriting output files …')
    write_geojson(sub_gdf,  'arcgis_substations_redday.geojson')
    write_geojson(edge_gdf, 'arcgis_edges_redday.geojson')

    # ── Summary ──────────────────────────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ArcGIS Usage                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  1. Add Data → arcgis_substations_redday.geojson                     ║
║     Add Data → arcgis_edges_redday.geojson                           ║
║                                                                      ║
║  2. Symbolize → Graduated Colors → field: diff_h01 (or any hour)     ║
║     Colour scheme: Diverging Blue–White–Red                          ║
║       Blue  = tempo-shifting reduced load                            ║
║       Red   = tempo-shifting increased load                          ║
║                                                                      ║
║  3. Step through hours:                                              ║
║     • Manual: change the symbolization field to diff_h01…diff_h24   ║
║     • Range Slider: Map tab → Range → Range Field → diff_h01…h24    ║
║     • Time Slider: pivot to long format first via                    ║
║         Data Engineering → Transpose Fields → diff_h01…diff_h24     ║
║         then enable Time on the resulting timestamp column           ║
║                                                                      ║
║  4. Both layers share substation_id — join them if needed.           ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
