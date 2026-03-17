#!/usr/bin/env python3
"""
make_arcgis_geojson.py
======================
Produces two ArcGIS-ready GeoJSON files in LONG FORMAT from the
distribution network shapefiles and load-profile CSVs.

Long format means one feature per (substation × hour) = 504 rows each.
This lets ArcGIS's built-in Time Slider drive the animation directly.

OUTPUT
------
output/arcgis_substations_redday_long.geojson
    504 Point features  (21 substations × 24 hours).
    Attributes:
        substation_id   — region folder name (e.g. "121579")
        timestamp       — ISO-8601 datetime "2014-07-03T01:00:00"
                          → used as the Time Slider field in ArcGIS
        hour            — integer 1–24 (convenience field)
        diff_kw         — Tempo-shifted minus Control load (kWh)
                          aggregated across all HID meters in this
                          substation for this hour of the Red Day.

output/arcgis_edges_redday_long.geojson
    504 MultiLineString features  (21 substations × 24 hours).
    Each feature contains ALL edges belonging to one substation at one
    hour-step, so the full tree network glows together as a unit.
    Same attributes as the substations file (timestamp, diff_kw, …).

CRS: EPSG:4326 (WGS84) — required for GeoJSON.

ARCGIS TIME SLIDER SETUP
-------------------------
  1. Add Data → select both .geojson files
  2. Right-click layer → Properties → Time tab
       Enable time on this layer: YES
       Time Field: timestamp
       Time Format: ISO 8601 (auto-detected)
  3. Repeat for the other layer; link both to the same time extent.
  4. Map tab → Time group → click the clock icon to open Time Slider.
  5. Symbolize both layers on "diff_kw" with a diverging ramp:
         Blue  = tempo-shifting reduced load
         Red   = tempo-shifting increased load
  6. Hit Play — substations and their network trees animate hour by hour.
"""

import os
import zipfile
import warnings
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString

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


def normalize_hid(x):
    try:    return str(int(float(x)))
    except: return None


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
    nodes = nodes.to_crs('EPSG:4326')
    edges = edges.to_crs('EPSG:4326')

    print(f'  {len(nodes):,} nodes  |  {len(edges):,} edges  |  '
          f'{nodes["region"].nunique()} substations')
    return nodes, edges


# ─── Step 3: Compute per-substation hourly diff (Red Day) ────────────────────
def compute_diff(nodes):
    """
    Returns a dict: diff[region][hour] = float (kWh)
    diff = sum(tempo) - sum(ctrl) across all HID meters in that substation.
    """
    print('\nLoading CSVs …')
    ctrl_df  = pd.read_csv(CONTROL_CSV)
    tempo_df = pd.read_csv(TEMPO_CSV)
    ctrl_df ['hid_key'] = ctrl_df ['hid'].apply(normalize_hid)
    tempo_df['hid_key'] = tempo_df['hid'].apply(normalize_hid)

    ctrl_red  = ctrl_df [ctrl_df ['date'] == RED_DAY]
    tempo_red = tempo_df[tempo_df['date'] == RED_DAY]
    print(f'  Control HIDs on Red Day: {ctrl_red ["hid_key"].nunique():,}')
    print(f'  Tempo   HIDs on Red Day: {tempo_red["hid_key"].nunique():,}')

    hh = nodes[nodes['label'] == 'H'][['hid', 'region']].copy()
    hh['hid_key'] = hh['hid'].apply(normalize_hid)
    hh = hh.dropna(subset=['hid_key'])

    def region_totals(csv_red):
        merged = hh.merge(csv_red[['hid_key'] + HOUR_COLS], on='hid_key', how='inner')
        return merged.groupby('region')[HOUR_COLS].sum()

    ctrl_tot  = region_totals(ctrl_red)
    tempo_tot = region_totals(tempo_red)

    all_regions = sorted(nodes['region'].unique())
    ctrl_tot  = ctrl_tot .reindex(all_regions, fill_value=0)
    tempo_tot = tempo_tot.reindex(all_regions, fill_value=0)

    diff_df = (tempo_tot - ctrl_tot).round(4)

    # Convert to nested dict: diff[region][hour_int] = float
    result = {}
    for region in all_regions:
        result[region] = {h: diff_df.loc[region, str(h)]
                          for h in range(1, 25)}

    vals = [v for hrs in result.values() for v in hrs.values()]
    print(f'  Diff range: {min(vals):.2f} … {max(vals):.2f} kWh  '
          f'(all {"≤0" if max(vals) <= 0 else "mixed"})')
    return result


# ─── Step 4: Build substation long-format GDF ─────────────────────────────────
def build_substations_long(nodes, diff):
    """
    504 Point features — one per (substation, hour).
    """
    subs = nodes[nodes['label'] == 'S'].copy()
    subs = subs[['region', 'geometry']].rename(columns={'region': 'substation_id'})

    rows = []
    for _, row in subs.iterrows():
        sid = row['substation_id']
        for h in range(1, 25):
            ts = f'{RED_DAY}T{h:02d}:00:00'
            rows.append({
                'substation_id': sid,
                'timestamp':     ts,
                'hour':          h,
                'diff_kw':       diff[sid][h],
                'geometry':      row['geometry'],
            })

    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
    print(f'\n  Substation long-format features: {len(gdf)}  '
          f'(expected {len(subs)*24} = {len(subs)} substations × 24 hours)')
    return gdf


# ─── Step 5: Build edge long-format GDF ──────────────────────────────────────
def build_edges_long(edges, diff):
    """
    504 MultiLineString features — one per (substation, hour).
    All edges for a substation are merged into a single MultiLineString
    so the full tree can be coloured/animated as one unit per time step.
    """
    # Collect individual LineStrings per substation
    sub_lines = {}
    for region, grp in edges.groupby('region'):
        lines = []
        for geom in grp.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                lines.append(geom)
            elif geom.geom_type == 'MultiLineString':
                lines.extend(geom.geoms)
        sub_lines[region] = lines

    regions = sorted(diff.keys())
    rows = []
    for sid in regions:
        multi = MultiLineString(sub_lines.get(sid, []))
        for h in range(1, 25):
            ts = f'{RED_DAY}T{h:02d}:00:00'
            rows.append({
                'substation_id': sid,
                'timestamp':     ts,
                'hour':          h,
                'diff_kw':       diff[sid][h],
                'geometry':      multi,
            })

    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
    print(f'  Edge long-format features:        {len(gdf)}  '
          f'(expected {len(regions)*24} = {len(regions)} substations × 24 hours)')
    return gdf


# ─── Step 6: Write GeoJSON ────────────────────────────────────────────────────
def write_geojson(gdf, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    # 6 decimal places ≈ 11 cm precision — more than adequate for visualization
    # and keeps file size within GitHub's 100 MB limit.
    gdf.to_file(path, driver='GeoJSON', COORDINATE_PRECISION=6)
    size_kb = os.path.getsize(path) // 1024
    print(f'  Wrote → {path}  ({len(gdf):,} features, {size_kb:,} KB)')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=== ArcGIS GeoJSON Export — Long Format  (Red Day Diff) ===')
    print(f'    {RED_DAY}   |   diff_kw = Tempo-shifted − Control\n')

    extract_zip()

    print('\nLoading shapefiles …')
    nodes, edges = load_network()

    diff = compute_diff(nodes)

    print('\nBuilding long-format layers …')
    sub_gdf  = build_substations_long(nodes, diff)
    edge_gdf = build_edges_long(edges, diff)

    print('\nWriting output files …')
    write_geojson(sub_gdf,  'arcgis_substations_redday_long.geojson')
    write_geojson(edge_gdf, 'arcgis_edges_redday_long.geojson')

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ArcGIS Time Slider Setup                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  1. Add Data → arcgis_substations_redday_long.geojson               ║
║     Add Data → arcgis_edges_redday_long.geojson                     ║
║                                                                      ║
║  2. Right-click each layer → Properties → Time tab                  ║
║       Enable time on this layer: YES                                 ║
║       Time Field: timestamp                                          ║
║       (ArcGIS auto-detects ISO-8601 format)                         ║
║                                                                      ║
║  3. Map tab → Time group → open Time Slider                          ║
║     Both layers share the same 24 hourly steps (01:00 … 24:00).     ║
║                                                                      ║
║  4. Symbolize → Graduated Colors → Field: diff_kw                   ║
║     Colour scheme: Diverging Blue–White–Red                          ║
║       Blue  = tempo-shifting reduced load (all values here ≤ 0)     ║
║       Red   = tempo-shifting increased load                          ║
║                                                                      ║
║  5. Hit Play.  Each step = 1 hour.  Substations and their full       ║
║     tree networks change colour together as diff_kw varies.          ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
