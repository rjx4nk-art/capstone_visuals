#!/usr/bin/env python3
"""
Distribution Network Load Visualization
========================================
Produces:
  output/
    01_red_day_animation.mp4        — control vs tempo, red day (24 hrs, side-by-side)
    02_daily_totals_animation.mp4   — daily total load comparison (7 frames, side-by-side)
    03_snapshot_peak.png            — static: peak-load hour on the red day
    04_snapshot_offpeak.png         — static: off-peak hour on the red day
    05_difference_map.png           — tempo minus control at peak hour (diverging colormap)
    06_substation_profiles.png      — 24h load curves per substation, red day
"""

import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase
import imageio
from PIL import Image
import io

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = '/home/user/capstone_visuals'
ZIP_PATH    = os.path.join(BASE_DIR, 'dist_net.zip')
CONTROL_CSV = os.path.join(BASE_DIR, 'control_profile_131.csv')
TEMPO_CSV   = os.path.join(BASE_DIR, 'tempo_shifted_profile_131.csv')
EXTRACT_DIR = '/tmp/dist_net'
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Visual constants ─────────────────────────────────────────────────────────
BG_COLOR    = '#06090f'    # deep navy-black background
PANEL_SEP   = '#10192a'    # slightly lighter for dividers
TEXT_COLOR  = '#c8d8e8'    # cool-white text
DIM_COLOR   = '#2a3f55'    # muted blue for borders / grid
CMAP_LOAD   = mcolors.LinearSegmentedColormap.from_list(
    'load', ['#00e676', '#ffe000', '#ff1744'], N=512)
CMAP_DIFF   = mcolors.LinearSegmentedColormap.from_list(
    'diff', ['#1565c0', '#e0e0e0', '#c62828'], N=512)

# Matplotlib style
plt.rcParams.update({
    'figure.facecolor':  BG_COLOR,
    'axes.facecolor':    BG_COLOR,
    'text.color':        TEXT_COLOR,
    'axes.labelcolor':   TEXT_COLOR,
    'xtick.color':       TEXT_COLOR,
    'ytick.color':       TEXT_COLOR,
    'axes.edgecolor':    DIM_COLOR,
    'font.family':       'monospace',
    'font.size':         9,
})

HOUR_COLS = [str(i) for i in range(1, 25)]

DAYS      = ['2014-07-01','2014-07-02','2014-07-03','2014-07-04',
             '2014-07-05','2014-07-06','2014-07-07']
DAY_COLOR = {'2014-07-01':'blue','2014-07-02':'white','2014-07-03':'red',
             '2014-07-04':'blue','2014-07-05':'blue','2014-07-06':'blue',
             '2014-07-07':'blue'}
DAY_LABEL = {
    '2014-07-01': 'Day 1  Tue Jul 1  [blue]',
    '2014-07-02': 'Day 2  Wed Jul 2  [white]',
    '2014-07-03': 'Day 3  Thu Jul 3  [red]',
    '2014-07-04': 'Day 4  Fri Jul 4  [blue]',
    '2014-07-05': 'Day 5  Sat Jul 5  [blue]',
    '2014-07-06': 'Day 6  Sun Jul 6  [blue]',
    '2014-07-07': 'Day 7  Mon Jul 7  [blue]',
}
RED_DAY  = '2014-07-03'


# ─── Step 1: Extract zip ──────────────────────────────────────────────────────
def extract_zip():
    flag = os.path.join(EXTRACT_DIR, '.done')
    if not os.path.exists(flag):
        print('Extracting network data…')
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(EXTRACT_DIR)
        open(flag, 'w').close()
    else:
        print('Network data already extracted.')


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
    print(f'  {len(nodes):,} nodes  |  {len(edges):,} edges  |  '
          f'{nodes["region"].nunique()} substations')
    return nodes, edges


def normalize_hid(x):
    try:    return str(int(float(x)))
    except: return None


# ─── Step 3: Load CSVs ────────────────────────────────────────────────────────
def load_csvs():
    def _load(path):
        df = pd.read_csv(path)
        df['hid_key'] = df['hid'].apply(normalize_hid)
        return df
    ctrl  = _load(CONTROL_CSV)
    tempo = _load(TEMPO_CSV)
    print(f'  CSV: {ctrl["hid_key"].nunique()} unique HIDs, '
          f'{ctrl["date"].nunique()} days')
    return ctrl, tempo


# ─── Step 4: Compute loads ────────────────────────────────────────────────────
def compute_loads(nodes, csv_df):
    """Return nested dict  loads[date][hour] = {region: total_kW}"""
    h = nodes[nodes['label'] == 'H'].copy()
    h['hid_key'] = h['hid'].apply(normalize_hid)
    h = h[['hid_key', 'region']].dropna(subset=['hid_key'])

    result = {}
    for date, day_df in csv_df.groupby('date'):
        merged = h.merge(day_df[['hid_key'] + HOUR_COLS], on='hid_key', how='inner')
        result[date] = {}
        for hr in HOUR_COLS:
            result[date][int(hr)] = merged.groupby('region')[hr].sum().to_dict()
    return result


def daily_totals(loads):
    """Return  totals[date] = {region: sum_over_24h}"""
    totals = {}
    for date, hours in loads.items():
        per_region = {}
        for hr_dict in hours.values():
            for reg, v in hr_dict.items():
                per_region[reg] = per_region.get(reg, 0.0) + v
        totals[date] = per_region
    return totals


def global_vmin_vmax(*load_dicts):
    vals = []
    for ld in load_dicts:
        for h_dict in ld.values():
            for v in h_dict.values():
                vals.extend(v.values())
    return min(vals), max(vals)


def global_daily_vmin_vmax(*total_dicts):
    vals = []
    for td in total_dicts:
        for v in td.values():
            vals.extend(v.values())
    return min(vals), max(vals)


# ─── Step 5: Pre-compute edge geometry segments ───────────────────────────────
def build_edge_segments(edges):
    """Return  {region: list_of_coord_arrays}"""
    seg_map = {}
    for region, grp in edges.groupby('region'):
        segs = []
        for geom in grp.geometry:
            if geom is None: continue
            if geom.geom_type == 'LineString':
                segs.append(np.array(geom.coords))
            elif geom.geom_type == 'MultiLineString':
                for part in geom.geoms:
                    segs.append(np.array(part.coords))
        seg_map[region] = segs
    return seg_map


def extract_substations(nodes):
    """Return GeoDataFrame of 'S' nodes (one per region)."""
    s = nodes[nodes['label'] == 'S'].copy()
    s['x'] = s.geometry.x
    s['y'] = s.geometry.y
    return s.reset_index(drop=True)


def map_extent(nodes, pad_frac=0.04):
    xs = nodes.geometry.x
    ys = nodes.geometry.y
    dx = xs.max() - xs.min()
    dy = ys.max() - ys.min()
    return (xs.min() - dx*pad_frac, xs.max() + dx*pad_frac,
            ys.min() - dy*pad_frac, ys.max() + dy*pad_frac)


# ─── Step 6: Rendering primitives ────────────────────────────────────────────
GLOW_LAYERS = [(10.0, 0.018), (5.0, 0.07), (2.5, 0.18), (1.0, 0.85)]

def draw_glow_edges(ax, segs, color, lw=0.7):
    if not segs:
        return
    for width_mult, alpha in GLOW_LAYERS:
        lc = LineCollection(segs, colors=[color], linewidths=lw*width_mult,
                            alpha=alpha, capstyle='round', joinstyle='round',
                            zorder=3)
        ax.add_collection(lc)


def draw_substation(ax, x, y, color):
    """Glowing substation node: outer halo → inner bright dot."""
    for s, a in [(280, 0.04), (120, 0.12), (45, 0.35)]:
        ax.scatter(x, y, s=s, color=color, alpha=a, zorder=7, linewidths=0)
    ax.scatter(x, y, s=18, color='white', alpha=0.95, zorder=8, linewidths=0)
    ax.scatter(x, y, s=7,  color=color,   alpha=1.00, zorder=9, linewidths=0)


def setup_map_ax(ax, xlim, ylim):
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect('equal')
    ax.axis('off')

    # Subtle lat/lon graticule
    for lon in np.arange(-76.1, -75.5, 0.1):
        ax.axvline(lon, color=DIM_COLOR, lw=0.35, alpha=0.5, zorder=1)
    for lat in np.arange(37.1, 37.7, 0.1):
        ax.axhline(lat, color=DIM_COLOR, lw=0.35, alpha=0.5, zorder=1)


def add_colorbar(fig, ax_or_pos, norm, label, cmap=CMAP_LOAD):
    cax = fig.add_axes(ax_or_pos)
    cb  = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label(label, color=TEXT_COLOR, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cax.set_facecolor(BG_COLOR)
    return cax


# ─── Step 7: Render a single side-by-side frame ───────────────────────────────
def render_side_by_side(fig, ax_ctrl, ax_tempo,
                        seg_map, substations,
                        loads_ctrl_frame, loads_tempo_frame,
                        norm, xlim, ylim,
                        title_ctrl, title_tempo, main_title):
    """
    Fill two axes with the glowing network.
    loads_ctrl_frame / loads_tempo_frame : {region: load_val}
    """
    for ax, loads_frame, panel_title in [
            (ax_ctrl,  loads_ctrl_frame,  title_ctrl),
            (ax_tempo, loads_tempo_frame, title_tempo)]:
        ax.cla()
        setup_map_ax(ax, xlim, ylim)

        for region, segs in seg_map.items():
            val   = loads_frame.get(region, 0.0)
            color = CMAP_LOAD(norm(val))
            draw_glow_edges(ax, segs, color)

        for _, row in substations.iterrows():
            val   = loads_frame.get(row['region'], 0.0)
            color = CMAP_LOAD(norm(val))
            draw_substation(ax, row['x'], row['y'], color)

        ax.set_title(panel_title, color=TEXT_COLOR, fontsize=9, pad=5,
                     loc='center')

    fig.suptitle(main_title, color='white', fontsize=11, y=0.97,
                 fontweight='bold')


def fig_to_rgb(fig):
    """Convert a matplotlib figure to an H×W×3 uint8 array (even dimensions)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    arr = np.array(img)
    # libx264 requires even width and height
    h, w = arr.shape[:2]
    if h % 2: arr = arr[:-1, :, :]
    if w % 2: arr = arr[:, :-1, :]
    return arr


# ─── Output 1: Red-day animation (24 frames) ─────────────────────────────────
def make_red_day_animation(seg_map, substations, loads_ctrl, loads_tempo,
                           vmin, vmax, xlim, ylim):
    print('\n[1/6] Rendering red-day animation (24 frames × 2 panels)…')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(18, 8.5), facecolor=BG_COLOR)
    # layout: [ctrl_map | tempo_map | colorbar]
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            width_ratios=[1, 1, 0.045],
                            left=0.01, right=0.95,
                            top=0.91, bottom=0.04,
                            wspace=0.04)
    ax_ctrl  = fig.add_subplot(gs[0])
    ax_tempo = fig.add_subplot(gs[1])
    cax      = fig.add_subplot(gs[2])
    cb = ColorbarBase(cax, cmap=CMAP_LOAD, norm=norm, orientation='vertical')
    cb.set_label('Aggregated load (kWh)', color=TEXT_COLOR, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cax.set_facecolor(BG_COLOR)

    # bottom legend bar
    fig.text(0.02, 0.01,
             '● Substation node    — Distribution network edges    '
             'Colour: green = low load  →  red = high load',
             color=DIM_COLOR, fontsize=7.5, va='bottom')

    frames = []
    for hr in range(1, 25):
        lc = loads_ctrl [RED_DAY][hr]
        lt = loads_tempo[RED_DAY][hr]
        render_side_by_side(
            fig, ax_ctrl, ax_tempo,
            seg_map, substations, lc, lt, norm, xlim, ylim,
            title_ctrl  = f'CONTROL  —  {RED_DAY}  hour {hr:02d}:00',
            title_tempo = f'TEMPO-SHIFTED  —  {RED_DAY}  hour {hr:02d}:00',
            main_title  = f'Distribution Network Load  |  Red Day ({RED_DAY})  '
                          f'|  Hour {hr:02d}:00 – {hr:02d}:59'
        )
        frames.append(fig_to_rgb(fig))
        print(f'  hour {hr:02d}', end='\r')

    plt.close(fig)

    out = os.path.join(OUTPUT_DIR, '01_red_day_animation.mp4')
    imageio.mimwrite(out, frames, fps=4, quality=8, macro_block_size=1)
    print(f'\n  Saved → {out}')


# ─── Output 2: Daily-totals animation (7 frames) ─────────────────────────────
def make_daily_animation(seg_map, substations, loads_ctrl, loads_tempo,
                         xlim, ylim):
    print('\n[2/6] Rendering daily-totals animation (7 frames)…')
    tot_ctrl  = daily_totals(loads_ctrl)
    tot_tempo = daily_totals(loads_tempo)
    vmin, vmax = global_daily_vmin_vmax(tot_ctrl, tot_tempo)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(18, 8.5), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            width_ratios=[1, 1, 0.045],
                            left=0.01, right=0.95,
                            top=0.91, bottom=0.04,
                            wspace=0.04)
    ax_ctrl  = fig.add_subplot(gs[0])
    ax_tempo = fig.add_subplot(gs[1])
    cax      = fig.add_subplot(gs[2])
    cb = ColorbarBase(cax, cmap=CMAP_LOAD, norm=norm, orientation='vertical')
    cb.set_label('Daily total load (kWh)', color=TEXT_COLOR, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cax.set_facecolor(BG_COLOR)

    fig.text(0.02, 0.01,
             'Each frame = one calendar day  |  Colour = daily total aggregated load per substation',
             color=DIM_COLOR, fontsize=7.5, va='bottom')

    frames = []
    for date in DAYS:
        tc = tot_ctrl [date]
        tt = tot_tempo[date]
        render_side_by_side(
            fig, ax_ctrl, ax_tempo,
            seg_map, substations, tc, tt, norm, xlim, ylim,
            title_ctrl  = f'CONTROL  —  {DAY_LABEL[date]}',
            title_tempo = f'TEMPO-SHIFTED  —  {DAY_LABEL[date]}',
            main_title  = f'Daily Total Load  |  {DAY_LABEL[date]}'
        )
        frames.append(fig_to_rgb(fig))
        print(f'  {date}')

    plt.close(fig)

    out = os.path.join(OUTPUT_DIR, '02_daily_totals_animation.mp4')
    imageio.mimwrite(out, frames, fps=1, quality=8, macro_block_size=1)
    print(f'  Saved → {out}')


# ─── Output 3 & 4: Static snapshots (peak / off-peak on red day) ─────────────
def make_snapshots(seg_map, substations, loads_ctrl, loads_tempo,
                   vmin, vmax, xlim, ylim):
    print('\n[3–4/6] Rendering peak / off-peak snapshots…')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Find peak and off-peak hour for control on red day
    hour_totals = {hr: sum(loads_ctrl[RED_DAY][hr].values())
                   for hr in range(1, 25)}
    peak_hr    = max(hour_totals, key=hour_totals.get)
    offpeak_hr = min(hour_totals, key=hour_totals.get)
    print(f'  Peak hour: {peak_hr:02d}:00  |  Off-peak: {offpeak_hr:02d}:00')

    for label, hr, fname in [
            ('PEAK LOAD', peak_hr,    '03_snapshot_peak.png'),
            ('OFF-PEAK LOAD', offpeak_hr, '04_snapshot_offpeak.png')]:

        fig = plt.figure(figsize=(18, 8.5), facecolor=BG_COLOR)
        gs  = gridspec.GridSpec(1, 3, figure=fig,
                                width_ratios=[1, 1, 0.045],
                                left=0.01, right=0.95,
                                top=0.91, bottom=0.07,
                                wspace=0.04)
        ax_ctrl  = fig.add_subplot(gs[0])
        ax_tempo = fig.add_subplot(gs[1])
        cax      = fig.add_subplot(gs[2])
        cb = ColorbarBase(cax, cmap=CMAP_LOAD, norm=norm, orientation='vertical')
        cb.set_label('Aggregated load (kWh)', color=TEXT_COLOR, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
        cax.set_facecolor(BG_COLOR)

        render_side_by_side(
            fig, ax_ctrl, ax_tempo,
            seg_map, substations,
            loads_ctrl [RED_DAY][hr],
            loads_tempo[RED_DAY][hr],
            norm, xlim, ylim,
            title_ctrl  = f'CONTROL  —  {RED_DAY}  hour {hr:02d}:00',
            title_tempo = f'TEMPO-SHIFTED  —  {RED_DAY}  hour {hr:02d}:00',
            main_title  = f'Distribution Network Load  |  {label}  '
                          f'|  {RED_DAY}  hour {hr:02d}:00'
        )

        out = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=BG_COLOR)
        plt.close(fig)
        print(f'  Saved → {out}')

    return peak_hr


# ─── Output 5: Difference map (tempo − control at peak hour) ─────────────────
def make_difference_map(seg_map, substations, loads_ctrl, loads_tempo,
                        peak_hr, xlim, ylim):
    print('\n[5/6] Rendering difference map (tempo − control at peak hour)…')

    lc = loads_ctrl [RED_DAY][peak_hr]
    lt = loads_tempo[RED_DAY][peak_hr]

    all_diffs = [lt.get(r, 0) - lc.get(r, 0) for r in seg_map]
    abs_max   = max(abs(v) for v in all_diffs) or 1.0
    norm      = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    fig = plt.figure(figsize=(10, 9), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            width_ratios=[1, 0.05],
                            left=0.03, right=0.93,
                            top=0.90, bottom=0.06,
                            wspace=0.06)
    ax  = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    setup_map_ax(ax, xlim, ylim)

    for region, segs in seg_map.items():
        diff  = lt.get(region, 0) - lc.get(region, 0)
        color = CMAP_DIFF(norm(diff))
        draw_glow_edges(ax, segs, color)

    for _, row in substations.iterrows():
        diff  = lt.get(row['region'], 0) - lc.get(row['region'], 0)
        color = CMAP_DIFF(norm(diff))
        draw_substation(ax, row['x'], row['y'], color)

    cb = ColorbarBase(cax, cmap=CMAP_DIFF, norm=norm, orientation='vertical')
    cb.set_label('Δ load: tempo − control (kWh)', color=TEXT_COLOR, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cax.set_facecolor(BG_COLOR)

    fig.suptitle(f'Load Difference  |  Tempo-Shifted − Control  '
                 f'|  {RED_DAY}  hour {peak_hr:02d}:00\n'
                 'Blue = tempo-shifting reduces load  '
                 '|  Red = tempo-shifting increases load',
                 color='white', fontsize=10, fontweight='bold', y=0.97)

    out = os.path.join(OUTPUT_DIR, '05_difference_map.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  Saved → {out}')


# ─── Output 6: Substation 24h load profiles (small multiples) ────────────────
def make_profile_plots(substations, loads_ctrl, loads_tempo):
    print('\n[6/6] Rendering 24h substation load profiles (small multiples)…')
    regions   = sorted(substations['region'].tolist())
    n_regions = len(regions)
    cols      = 4
    rows      = (n_regions + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*3.2),
                             facecolor=BG_COLOR)
    axes_flat = axes.flatten()

    hours = list(range(1, 25))

    for i, region in enumerate(regions):
        ax = axes_flat[i]
        ax.set_facecolor('#0c1622')
        ax.spines[:].set_color(DIM_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=6.5)

        ctrl_vals  = [loads_ctrl [RED_DAY][h].get(region, 0) for h in hours]
        tempo_vals = [loads_tempo[RED_DAY][h].get(region, 0) for h in hours]

        ax.plot(hours, ctrl_vals,  color='#00e676', lw=1.5, label='Control',
                zorder=3)
        ax.plot(hours, tempo_vals, color='#ff8c00', lw=1.5, label='Tempo-shifted',
                linestyle='--', zorder=4)
        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=[t > c for t, c in zip(tempo_vals, ctrl_vals)],
                        alpha=0.12, color='#ff4444', zorder=2)
        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=[t < c for t, c in zip(tempo_vals, ctrl_vals)],
                        alpha=0.12, color='#4444ff', zorder=2)

        ax.set_title(f'Substation {region}', color=TEXT_COLOR,
                     fontsize=8, pad=3)
        ax.set_xlabel('Hour', color=TEXT_COLOR, fontsize=6.5)
        ax.set_ylabel('Load (kWh)', color=TEXT_COLOR, fontsize=6.5)
        ax.set_xlim(1, 24)
        ax.set_xticks(range(1, 25, 4))
        ax.grid(color=DIM_COLOR, lw=0.4, alpha=0.7)

    # Turn off unused axes
    for j in range(n_regions, len(axes_flat)):
        axes_flat[j].axis('off')

    # Shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color='#00e676', lw=2,  label='Control'),
        Line2D([0],[0], color='#ff8c00', lw=2, linestyle='--',
               label='Tempo-shifted'),
    ]
    fig.legend(handles=handles, loc='lower right', fontsize=9,
               facecolor='#0c1622', edgecolor=DIM_COLOR,
               labelcolor=TEXT_COLOR, ncol=2)

    fig.suptitle(f'24-Hour Load Profiles per Substation  |  Red Day ({RED_DAY})\n'
                 'Green = Control  |  Orange dashed = Tempo-shifted  |  '
                 'Red fill = tempo > ctrl  |  Blue fill = tempo < ctrl',
                 color='white', fontsize=11, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    out = os.path.join(OUTPUT_DIR, '06_substation_profiles.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  Saved → {out}')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=== Distribution Network Load Visualization ===\n')

    extract_zip()

    print('\nLoading network shapefiles…')
    nodes, edges = load_network()

    print('\nLoading CSV load profiles…')
    ctrl_df, tempo_df = load_csvs()

    print('\nComputing substation-level hourly loads…')
    loads_ctrl  = compute_loads(nodes, ctrl_df)
    loads_tempo = compute_loads(nodes, tempo_df)

    vmin, vmax = global_vmin_vmax(loads_ctrl, loads_tempo)
    print(f'  Global load range: {vmin:.1f} – {vmax:.1f} kWh')

    print('\nPre-computing edge segments…')
    seg_map    = build_edge_segments(edges)
    substations = extract_substations(nodes)

    x0, x1, y0, y1 = map_extent(nodes)
    xlim, ylim = (x0, x1), (y0, y1)
    print(f'  Extent: lon [{x0:.3f}, {x1:.3f}]  lat [{y0:.3f}, {y1:.3f}]')

    # ── Generate all outputs ─────────────────────────────────────────────────
    make_red_day_animation(seg_map, substations, loads_ctrl, loads_tempo,
                           vmin, vmax, xlim, ylim)
    make_daily_animation   (seg_map, substations, loads_ctrl, loads_tempo,
                            xlim, ylim)
    peak_hr = make_snapshots(seg_map, substations, loads_ctrl, loads_tempo,
                             vmin, vmax, xlim, ylim)
    make_difference_map    (seg_map, substations, loads_ctrl, loads_tempo,
                            peak_hr, xlim, ylim)
    make_profile_plots     (substations, loads_ctrl, loads_tempo)

    print('\n=== Done. All outputs in:', OUTPUT_DIR, '===')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        print(f'  {f:45s}  {os.path.getsize(path)//1024:>6} KB')


if __name__ == '__main__':
    main()
