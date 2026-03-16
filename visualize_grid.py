#!/usr/bin/env python3
"""
Distribution Network Load Visualization — Publication Quality
=============================================================
Produces:
  output/
    01_diff_animation.mp4      — Animated tempo-minus-control difference map
                                 7 days × 24 hours (168 frames)
                                 CartoDB Positron basemap, large substation
                                 markers, thin tree-network spindles
    02_substation_profiles.png — 24h load profiles per substation, Red Day
                                 (the day tempo-shifting was active)
    03_system_profiles.png     — System-total 24h profiles, 3 panels:
                                 Red Day / White Day / Blue-day average

Data: data/control_profile.csv and data/tempo_shifted_profile.csv
      28,195 unique HIDs across 21 substations, 7 days (Jul 1-7 2014)
"""

import os
import zipfile
import warnings
import io

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
import imageio
from PIL import Image
import contextily as ctx

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = '/home/user/capstone_visuals'
ZIP_PATH    = os.path.join(BASE_DIR, 'dist_net.zip')
CONTROL_CSV = os.path.join(BASE_DIR, 'data', 'control_profile.csv')
TEMPO_CSV   = os.path.join(BASE_DIR, 'data', 'tempo_shifted_profile.csv')
EXTRACT_DIR = '/tmp/dist_net'
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Day definitions ──────────────────────────────────────────────────────────
HOUR_COLS = [str(i) for i in range(1, 25)]
DAYS      = ['2014-07-01','2014-07-02','2014-07-03',
             '2014-07-04','2014-07-05','2014-07-06','2014-07-07']
DAY_TYPE  = {d: 'blue' for d in DAYS}
DAY_TYPE['2014-07-02'] = 'white'
DAY_TYPE['2014-07-03'] = 'red'
RED_DAY   = '2014-07-03'
WHITE_DAY = '2014-07-02'
BLUE_DAYS = [d for d in DAYS if DAY_TYPE[d] == 'blue']

DAY_LABEL = {
    '2014-07-01': 'Day 1  Tue 1 Jul  [Blue]',
    '2014-07-02': 'Day 2  Wed 2 Jul  [White]',
    '2014-07-03': 'Day 3  Thu 3 Jul  [Red — tempo active]',
    '2014-07-04': 'Day 4  Fri 4 Jul  [Blue]',
    '2014-07-05': 'Day 5  Sat 5 Jul  [Blue]',
    '2014-07-06': 'Day 6  Sun 6 Jul  [Blue]',
    '2014-07-07': 'Day 7  Mon 7 Jul  [Blue]',
}

# Publication-quality color palette (ColorBrewer / scientific)
CTRL_COLOR  = '#2166AC'   # Dark blue — control scenario
TEMPO_COLOR = '#D6604D'   # Warm red-orange — tempo-shifted scenario
FILL_POS    = '#f4a582'   # Light coral  (tempo > ctrl)
FILL_NEG    = '#92c5de'   # Light sky-blue (tempo < ctrl)
EDGE_COLOR  = '#707070'   # Neutral gray for network edges
CMAP_DIFF   = plt.cm.RdBu_r  # Diverging: blue=decrease, red=increase

# Matplotlib global style — clean, academic
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          10,
    'axes.titlesize':     12,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.grid':          True,
    'grid.color':         '#e0e0e0',
    'grid.linewidth':     0.6,
    'text.color':         '#222222',
    'axes.labelcolor':    '#222222',
    'xtick.color':        '#444444',
    'ytick.color':        '#444444',
    'axes.edgecolor':     '#cccccc',
})


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


# ─── Step 5: Pre-compute edge geometry segments (in projected CRS) ────────────
def build_edge_segments(edges):
    """Return  {region: list_of_coord_arrays}"""
    seg_map = {}
    for region, grp in edges.groupby('region'):
        segs = []
        for geom in grp.geometry:
            if geom is None:
                continue
            if geom.geom_type == 'LineString':
                segs.append(np.array(geom.coords))
            elif geom.geom_type == 'MultiLineString':
                for part in geom.geoms:
                    segs.append(np.array(part.coords))
        seg_map[region] = segs
    return seg_map


def extract_substations(nodes):
    s = nodes[nodes['label'] == 'S'].copy()
    s['x'] = s.geometry.x
    s['y'] = s.geometry.y
    return s.reset_index(drop=True)


# ─── Step 6: Fetch CartoDB Positron basemap tiles (once) ─────────────────────
def load_basemap(nodes_3857, pad_frac=0.08):
    """
    Fetch CartoDB Positron tiles for the network extent.
    Falls back to OpenStreetMap.Mapnik, then plain white.
    Returns (img_array, [west, east, south, north]) in EPSG:3857.
    """
    xs = nodes_3857.geometry.x
    ys = nodes_3857.geometry.y
    dx, dy = xs.max() - xs.min(), ys.max() - ys.min()
    w = xs.min() - dx * pad_frac
    e = xs.max() + dx * pad_frac
    s = ys.min() - dy * pad_frac
    n = ys.max() + dy * pad_frac

    for provider, name in [
        (ctx.providers.CartoDB.Positron,       'CartoDB Positron'),
        (ctx.providers.OpenStreetMap.Mapnik,   'OpenStreetMap Mapnik'),
    ]:
        try:
            img, ext = ctx.bounds2img(w, s, e, n, zoom=13, source=provider)
            print(f'  Basemap: {name}  ({img.shape[1]}×{img.shape[0]} px)')
            return img, list(ext)   # ext = (west, east, south, north)
        except Exception as ex:
            print(f'  {name} failed: {ex}')

    print('  Basemap: plain white (no tiles available)')
    return None, [w, e, s, n]


def map_xlim_ylim(nodes_3857, pad_frac=0.05):
    xs = nodes_3857.geometry.x
    ys = nodes_3857.geometry.y
    dx, dy = xs.max() - xs.min(), ys.max() - ys.min()
    return ((xs.min() - dx*pad_frac, xs.max() + dx*pad_frac),
            (ys.min() - dy*pad_frac, ys.max() + dy*pad_frac))


def setup_map_ax(ax, xlim, ylim, basemap_img, basemap_ext):
    ax.set_facecolor('white')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    if basemap_img is not None:
        ax.imshow(basemap_img, extent=basemap_ext, origin='upper',
                  zorder=0, interpolation='bilinear')


def fig_to_rgb(fig):
    """Convert a matplotlib figure to an H×W×3 uint8 array (even dimensions)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    if h % 2: arr = arr[:-1, :, :]
    if w % 2: arr = arr[:, :-1, :]
    return arr


# ─── Output 1: Animated difference map (7 days × 24 hours = 168 frames) ───────
def make_diff_animation(seg_map_3857, subs_3857, basemap_img, basemap_ext,
                        loads_ctrl, loads_tempo, xlim, ylim):
    print('\n[1/3] Building difference animation (168 frames)…')

    regions = list(seg_map_3857.keys())

    # Global diff range across all 168 frames → fixed colorbar
    all_diffs = []
    for date in DAYS:
        for hr in range(1, 25):
            lc = loads_ctrl[date][hr]
            lt = loads_tempo[date][hr]
            for r in regions:
                all_diffs.append(lt.get(r, 0) - lc.get(r, 0))
    abs_max = max(abs(v) for v in all_diffs) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # Pre-compute all edge segments as a single LineCollection per region
    # We'll just store the segs; the collection is rebuilt each frame because
    # the colors change.
    all_segs = []
    for r in regions:
        all_segs.extend(seg_map_3857[r])

    # Figure layout: map | colorbar
    fig = plt.figure(figsize=(14, 9), facecolor='white')
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            width_ratios=[1, 0.03],
                            left=0.01, right=0.94,
                            top=0.90, bottom=0.04,
                            wspace=0.03)
    ax  = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    cb = ColorbarBase(cax, cmap=CMAP_DIFF, norm=norm, orientation='vertical')
    cb.set_label('Load difference: Tempo − Control (kWh)', fontsize=9)
    cb.ax.yaxis.set_tick_params(labelsize=8)
    cb.ax.set_facecolor('white')
    # Annotate extremes
    cax.text(0.5, 1.02, '▲ Tempo\nincreases\nload', transform=cax.transAxes,
             ha='center', va='bottom', fontsize=7, color='#b2182b')
    cax.text(0.5, -0.02, '▼ Tempo\nreduces\nload', transform=cax.transAxes,
             ha='center', va='top', fontsize=7, color='#2166ac')

    # Bottom caption
    fig.text(0.02, 0.01,
             'Network edges shown for topology only.  '
             'Substation circle size ∝ |Δload|.  '
             'Red = tempo-shifting increases load.  Blue = tempo-shifting reduces load.',
             fontsize=7.5, color='#555555', va='bottom')

    frames = []
    frame_n = 0
    for date in DAYS:
        day_type = DAY_TYPE[date]
        day_type_str = {'red': 'Red Day  (tempo-shifting active)',
                        'white': 'White Day  (baseline)',
                        'blue': 'Blue Day  (baseline)'}[day_type]
        badge_color = {'red': '#e6550d', 'white': '#636363', 'blue': '#3182bd'}[day_type]

        for hr in range(1, 25):
            lc = loads_ctrl[date][hr]
            lt = loads_tempo[date][hr]

            ax.cla()
            setup_map_ax(ax, xlim, ylim, basemap_img, basemap_ext)

            # ── Network edges: thin, neutral, topology-only ──────────────────
            lc_edges = LineCollection(
                all_segs,
                colors=EDGE_COLOR, linewidths=0.45, alpha=0.45,
                capstyle='round', joinstyle='round', zorder=2
            )
            ax.add_collection(lc_edges)

            # ── Substation dots: large, colored by difference ────────────────
            diff_vals = np.array([lt.get(r, 0) - lc.get(r, 0) for r in regions])
            colors    = CMAP_DIFF(norm(diff_vals))

            xs_s = subs_3857.set_index('region').loc[regions, 'x'].values
            ys_s = subs_3857.set_index('region').loc[regions, 'y'].values

            # Size proportional to |diff|, with a minimum for visibility
            max_diff = abs_max if abs_max > 0 else 1
            sizes = 120 + 300 * (np.abs(diff_vals) / max_diff)

            # Outer white halo for contrast against basemap
            ax.scatter(xs_s, ys_s, s=sizes + 80, c='white',
                       zorder=4, linewidths=0)
            # Colored circle
            ax.scatter(xs_s, ys_s, s=sizes, c=colors,
                       zorder=5, linewidths=1.0,
                       edgecolors='#444444')

            # ── Title ────────────────────────────────────────────────────────
            fig.suptitle(
                f'{DAY_LABEL[date]}   |   Hour {hr:02d}:00',
                fontsize=13, fontweight='bold', color='#111111',
                y=0.96
            )
            # Day-type badge
            ax.set_title(day_type_str, fontsize=10, color=badge_color,
                         pad=4, loc='left')

            frames.append(fig_to_rgb(fig))
            frame_n += 1
            if hr == 1 or hr % 6 == 0:
                print(f'  {date}  hour {hr:02d}  [{frame_n}/168]', end='\r')

    plt.close(fig)

    out = os.path.join(OUTPUT_DIR, '01_diff_animation.mp4')
    imageio.mimwrite(out, frames, fps=4, quality=9, macro_block_size=1)
    print(f'\n  Saved → {out}  ({len(frames)} frames)')


# ─── Output 2: Substation 24h profiles — Red Day small multiples ─────────────
def make_substation_profiles(substations, loads_ctrl, loads_tempo):
    """
    24-hour load profiles for each substation on the Red Day (2014-07-03),
    the only day when tempo-shifting was active.  Control (solid blue) vs
    Tempo-shifted (dashed red-orange).
    """
    print('\n[2/3] Rendering substation profiles (Red Day small multiples)…')

    regions   = sorted(substations['region'].tolist())
    n         = len(regions)
    cols      = 4
    rows      = (n + cols - 1) // cols
    hours     = list(range(1, 25))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4.8, rows * 3.4),
                             facecolor='white')
    axes_flat = axes.flatten()

    # Shared y-limits for all panels
    all_vals = []
    for r in regions:
        all_vals += [loads_ctrl[RED_DAY][h].get(r, 0) for h in hours]
        all_vals += [loads_tempo[RED_DAY][h].get(r, 0) for h in hours]
    ymax = max(all_vals) * 1.08

    for i, region in enumerate(regions):
        ax = axes_flat[i]
        ctrl_vals  = [loads_ctrl [RED_DAY][h].get(region, 0) for h in hours]
        tempo_vals = [loads_tempo[RED_DAY][h].get(region, 0) for h in hours]

        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=[t > c for t, c in zip(tempo_vals, ctrl_vals)],
                        color=FILL_POS, alpha=0.55, zorder=1,
                        label='_nolegend_')
        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=[t < c for t, c in zip(tempo_vals, ctrl_vals)],
                        color=FILL_NEG, alpha=0.55, zorder=1,
                        label='_nolegend_')
        ax.plot(hours, ctrl_vals,  color=CTRL_COLOR,  lw=1.8,
                solid_capstyle='round', zorder=3, label='Control')
        ax.plot(hours, tempo_vals, color=TEMPO_COLOR, lw=1.8,
                linestyle='--', dashes=(5, 2.5),
                solid_capstyle='round', zorder=4, label='Tempo-shifted')

        ax.set_title(f'Substation {region}', fontsize=9, fontweight='semibold',
                     color='#111111', pad=3)
        ax.set_xlim(1, 24)
        ax.set_ylim(0, ymax)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xticklabels(['01:00', '06:00', '12:00', '18:00', '24:00'],
                           fontsize=7.5)
        ax.yaxis.set_tick_params(labelsize=7.5)
        ax.set_xlabel('Hour of day', fontsize=8, labelpad=2)
        ax.set_ylabel('Load (kWh)', fontsize=8, labelpad=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(axis='y', color='#eeeeee', lw=0.7)
        ax.grid(axis='x', color='#eeeeee', lw=0.5)
        ax.set_facecolor('#fafafa')

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=CTRL_COLOR,  lw=2,   label='Control'),
        Line2D([0], [0], color=TEMPO_COLOR, lw=2,
               linestyle='--', dashes=(5, 2.5), label='Tempo-shifted'),
        mpatches.Patch(color=FILL_POS, alpha=0.7,
                       label='Tempo > Control (load increase)'),
        mpatches.Patch(color=FILL_NEG, alpha=0.7,
                       label='Tempo < Control (load reduction)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=4, fontsize=9, frameon=True,
               facecolor='white', edgecolor='#cccccc',
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f'24-Hour Substation Load Profiles  —  Red Day ({RED_DAY})\n'
        'The Red Day is the sole day on which the tempo-shifting tariff was active.',
        fontsize=13, fontweight='bold', y=1.02, color='#111111'
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = os.path.join(OUTPUT_DIR, '02_substation_profiles.png')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {out}')


# ─── Output 3: System-aggregate profiles — Red / White / Blue-average ─────────
def make_system_profiles(loads_ctrl, loads_tempo):
    """
    Aggregate load across all 21 substations, then plot 24h profiles for:
      Panel 1 — Red Day   (2014-07-03, tempo-shifting active)
      Panel 2 — White Day (2014-07-02, baseline peak-ish day)
      Panel 3 — Blue-day average (mean of all 5 blue baseline days)

    Each panel shows Control (solid) vs Tempo-shifted (dashed).
    """
    print('\n[3/3] Rendering system-aggregate profiles (3 panels)…')

    hours = list(range(1, 25))

    def system_series(loads, date_or_dates):
        """Sum across all substations per hour; average over multiple dates."""
        if isinstance(date_or_dates, str):
            dates = [date_or_dates]
        else:
            dates = list(date_or_dates)
        arr = np.array([
            [sum(loads[d][h].values()) for h in hours]
            for d in dates
        ])
        return arr.mean(axis=0)   # shape: (24,)

    # Build data for each panel
    panels = [
        {
            'label':      f'Red Day  ({RED_DAY})',
            'sublabel':   'Tempo-shifting active',
            'dates_ctrl':  RED_DAY,
            'dates_tempo': RED_DAY,
            'badge':      '#e6550d',
            'alpha_fill':  0.45,
        },
        {
            'label':      f'White Day  ({WHITE_DAY})',
            'sublabel':   'Baseline (no tempo-shifting)',
            'dates_ctrl':  WHITE_DAY,
            'dates_tempo': WHITE_DAY,
            'badge':      '#636363',
            'alpha_fill':  0.30,
        },
        {
            'label':      'Blue Days  (average)',
            'sublabel':   f'Mean of {len(BLUE_DAYS)} baseline days',
            'dates_ctrl':  BLUE_DAYS,
            'dates_tempo': BLUE_DAYS,
            'badge':      '#3182bd',
            'alpha_fill':  0.30,
        },
    ]

    # Shared y-axis scale
    all_vals = []
    for p in panels:
        all_vals += list(system_series(loads_ctrl,  p['dates_ctrl']))
        all_vals += list(system_series(loads_tempo, p['dates_tempo']))
    ymax = max(all_vals) * 1.08
    ymin = 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor='white',
                             sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.18,
                        wspace=0.08)

    for ax, p in zip(axes, panels):
        ctrl_vals  = system_series(loads_ctrl,  p['dates_ctrl'])
        tempo_vals = system_series(loads_tempo, p['dates_tempo'])

        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=tempo_vals > ctrl_vals,
                        color=FILL_POS, alpha=p['alpha_fill'],
                        zorder=1)
        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=tempo_vals < ctrl_vals,
                        color=FILL_NEG, alpha=p['alpha_fill'],
                        zorder=1)
        ax.plot(hours, ctrl_vals,  color=CTRL_COLOR,  lw=2.2,
                solid_capstyle='round', zorder=3, label='Control')
        ax.plot(hours, tempo_vals, color=TEMPO_COLOR, lw=2.2,
                linestyle='--', dashes=(6, 3),
                solid_capstyle='round', zorder=4, label='Tempo-shifted')

        # Peak annotation on Red Day
        if 'Red' in p['label']:
            peak_h = int(np.argmax(ctrl_vals)) + 1   # hours are 1-indexed
            peak_v = ctrl_vals[peak_h - 1]
            ax.annotate(f'Peak\n{peak_v/1e6:.2f} GWh',
                        xy=(peak_h, peak_v),
                        xytext=(peak_h - 4, peak_v * 0.93),
                        arrowprops=dict(arrowstyle='->', color='#555',
                                        lw=0.9),
                        fontsize=8, color='#333')

        ax.set_xlim(1, 24)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xticklabels(['01:00', '06:00', '12:00', '18:00', '24:00'],
                           fontsize=8.5)
        ax.set_xlabel('Hour of day', fontsize=9.5, labelpad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(axis='y', color='#eeeeee', lw=0.8)
        ax.grid(axis='x', color='#eeeeee', lw=0.5)
        ax.set_facecolor('#fafafa')

        ax.set_title(p['label'], fontsize=12, fontweight='bold',
                     color=p['badge'], pad=6)
        ax.text(0.5, 1.04, p['sublabel'],
                transform=ax.transAxes, ha='center',
                fontsize=9, color='#555555', style='italic')

    axes[0].set_ylabel('Total system load (kWh)', fontsize=10, labelpad=6)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=CTRL_COLOR,  lw=2.2, label='Control'),
        Line2D([0], [0], color=TEMPO_COLOR, lw=2.2,
               linestyle='--', dashes=(6, 3), label='Tempo-shifted'),
        mpatches.Patch(color=FILL_POS, alpha=0.7,
                       label='Tempo > Control (load increase)'),
        mpatches.Patch(color=FILL_NEG, alpha=0.7,
                       label='Tempo < Control (load reduction)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=4, fontsize=9.5, frameon=True,
               facecolor='white', edgecolor='#cccccc',
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        'System-Aggregate 24-Hour Load Profile  —  Control vs. Tempo-Shifted\n'
        '21 substations  |  28,195 households  |  July 2014  |  '
        'Blue-day panel = mean of 5 baseline days',
        fontsize=13, fontweight='bold', y=0.98, color='#111111'
    )

    out = os.path.join(OUTPUT_DIR, '03_system_profiles.png')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {out}')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=== Distribution Network Load Visualization — Publication Edition ===\n')

    extract_zip()

    print('\nLoading network shapefiles…')
    nodes, edges = load_network()

    print('\nLoading CSV load profiles…')
    ctrl_df, tempo_df = load_csvs()

    print('\nComputing substation-level hourly loads…')
    loads_ctrl  = compute_loads(nodes, ctrl_df)
    loads_tempo = compute_loads(nodes, tempo_df)

    # Reproject to Web Mercator (EPSG:3857) for basemap overlay
    print('\nReprojecting network to EPSG:3857…')
    nodes_3857 = nodes.to_crs('EPSG:3857')
    edges_3857 = edges.to_crs('EPSG:3857')

    seg_map_3857  = build_edge_segments(edges_3857)
    subs_3857     = extract_substations(nodes_3857)
    xlim, ylim    = map_xlim_ylim(nodes_3857)

    print('\nFetching basemap tiles…')
    basemap_img, basemap_ext = load_basemap(nodes_3857)

    print(f'\nMap extent (EPSG:3857):  '
          f'x [{xlim[0]:.0f}, {xlim[1]:.0f}]  '
          f'y [{ylim[0]:.0f}, {ylim[1]:.0f}]')

    # ── Generate outputs ─────────────────────────────────────────────────────
    make_diff_animation(seg_map_3857, subs_3857, basemap_img, basemap_ext,
                        loads_ctrl, loads_tempo, xlim, ylim)

    make_substation_profiles(subs_3857, loads_ctrl, loads_tempo)

    make_system_profiles(loads_ctrl, loads_tempo)

    print('\n=== Done.  Outputs in:', OUTPUT_DIR, '===')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        print(f'  {f:45s}  {os.path.getsize(path)//1024:>6} KB')


if __name__ == '__main__':
    main()
