#!/usr/bin/env python3
"""
Distribution Network Map — OSM Overlay
=======================================
Plots the distribution network on an OpenStreetMap basemap using geopandas
and contextily. Each substation's edges and node are drawn in a distinct color.

Usage in Google Colab:
  1. Upload dist_net.zip to your Colab session (Files panel → Upload).
  2. Run:  !python visualize_dist_net_map.py
     OR paste contents into a code cell and run it.

Output:
  /content/dist_net_map.png   — saved map image
  Console                     — total node and edge counts
"""

# ── Install dependencies (safe to re-run) ─────────────────────────────────────
import subprocess, sys
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'contextily', 'geopandas', 'matplotlib', 'shapely'],
    stdout=subprocess.DEVNULL
)

import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx

warnings.filterwarnings('ignore')

# ── Paths (Colab defaults) ─────────────────────────────────────────────────────
ZIP_PATH    = '/content/dist_net.zip'
EXTRACT_DIR = '/tmp/dist_net'
OUTPUT_PATH = '/content/dist_net_map.png'

# ── Step 1: Extract zip ────────────────────────────────────────────────────────
done_flag = os.path.join(EXTRACT_DIR, '.done')
if not os.path.exists(done_flag):
    print('Extracting dist_net.zip …')
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)
    open(done_flag, 'w').close()
else:
    print('Network data already extracted.')

# ── Step 2: Load shapefiles ────────────────────────────────────────────────────
print('\nLoading shapefiles …')
content_dir = os.path.join(EXTRACT_DIR, 'content', 'output')

all_nodes, all_edges = [], []

for folder in sorted(os.listdir(content_dir)):
    folder_path = os.path.join(content_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    node_shp = os.path.join(folder_path, f'{folder}-nodelist-HID.shp')
    edge_shp = os.path.join(folder_path, f'{folder}-edgelist.shp')

    if os.path.exists(node_shp):
        gdf = gpd.read_file(node_shp)
        gdf['substation'] = folder
        all_nodes.append(gdf)

    if os.path.exists(edge_shp):
        gdf = gpd.read_file(edge_shp)
        gdf['substation'] = folder
        all_edges.append(gdf)

nodes = gpd.GeoDataFrame(
    pd.concat(all_nodes, ignore_index=True),
    geometry='geometry', crs=all_nodes[0].crs
)
edges = gpd.GeoDataFrame(
    pd.concat(all_edges, ignore_index=True),
    geometry='geometry', crs=all_edges[0].crs
)

total_nodes = len(nodes)
total_edges = len(edges)
n_substations = nodes['substation'].nunique()

print(f'  Total nodes  : {total_nodes:,}')
print(f'  Total edges  : {total_edges:,}')
print(f'  Substations  : {n_substations}')

# ── Step 3: Extract substation nodes (label == 'S') ───────────────────────────
substation_nodes = nodes[nodes['label'] == 'S'].copy()

# ── Step 4: Reproject to Web Mercator (required for contextily OSM tiles) ──────
print('\nReprojecting to Web Mercator (EPSG:3857) …')
WEB_MERCATOR = 'EPSG:3857'
edges_wm  = edges.to_crs(WEB_MERCATOR)
subs_wm   = substation_nodes.to_crs(WEB_MERCATOR)

# ── Step 5: Assign a distinct color to each substation ────────────────────────
substation_ids = sorted(edges['substation'].unique())

# Build a palette large enough for any number of substations.
# Interleave tab20 and tab20b for maximum visual separation.
_t20  = plt.get_cmap('tab20')
_t20b = plt.get_cmap('tab20b')
palette = []
for i in range(20):
    palette.append(_t20(i))
    palette.append(_t20b(i))

color_map = {sid: palette[i % len(palette)]
             for i, sid in enumerate(substation_ids)}

# ── Step 6: Draw map ───────────────────────────────────────────────────────────
print('\nRendering map …')
# 16:9 widescreen — fits a slide without cropping
fig, ax = plt.subplots(figsize=(20, 11.25))

# Edges — one layer per substation
# Heavier linewidth so network branches read clearly at slide scale
for sid in substation_ids:
    sub_edges = edges_wm[edges_wm['substation'] == sid]
    if sub_edges.empty:
        continue
    sub_edges.plot(
        ax=ax,
        color=color_map[sid],
        linewidth=1.8,
        alpha=0.9,
        zorder=2
    )

# Substation nodes — star markers on top
for sid in substation_ids:
    sub_node = subs_wm[subs_wm['substation'] == sid]
    if sub_node.empty:
        continue
    sub_node.plot(
        ax=ax,
        color=color_map[sid],
        markersize=200,
        marker='*',
        zorder=4,
        edgecolors='white',
        linewidths=1.0
    )

# OSM basemap — slightly more transparent so colored edges pop
ctx.add_basemap(
    ax,
    crs=WEB_MERCATOR,
    source=ctx.providers.OpenStreetMap.Mapnik,
    zoom='auto',
    alpha=0.45,
    zorder=1
)

ax.set_title(
    'Distribution Network — Edges & Substation Nodes\n'
    f'Total nodes: {total_nodes:,}   |   Total edges: {total_edges:,}   |   '
    f'Substations: {n_substations}',
    fontsize=15,
    fontweight='bold',
    pad=16
)
ax.axis('off')

# Legend — spread across more columns to stay compact in widescreen layout
legend_handles = [
    mpatches.Patch(color=color_map[sid], label=f'Substation {sid}')
    for sid in substation_ids
]
ax.legend(
    handles=legend_handles,
    loc='lower left',
    fontsize=9,
    framealpha=0.85,
    ncol=3,
    title='Substations',
    title_fontsize=10
)

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nMap saved → {OUTPUT_PATH}')

# ── Summary ────────────────────────────────────────────────────────────────────
print('\n=== Network Totals ===')
print(f'  Total nodes  : {total_nodes:,}')
print(f'  Total edges  : {total_edges:,}')
