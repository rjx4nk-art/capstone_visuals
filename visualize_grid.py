#!/usr/bin/env python3
"""
Distribution Network Load Visualization — Publication Quality
=============================================================
Produces:
  output/
    01_diff_animation.mp4      — 168-frame animated difference map
                                 (7 days × 24 hours).  Each substation's
                                 tree glows/pulses in colour by Tempo−Control.
                                 Large fixed landmark dot marks each substation.
    02_substation_profiles.png — 24h load profiles per substation, Red Day
    03_system_profiles.png     — System-total 24h profiles:
                                 Red Day / White Day / Blue-day average

Visual language inherits the original dark-background glow aesthetic.
"""

import os, zipfile, warnings, io
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

# ─── Visual constants — dark glow aesthetic ───────────────────────────────────
BG_COLOR   = '#06090f'   # deep navy-black
DIM_COLOR  = '#1a2d42'   # muted blue for grid / graticule
TEXT_COLOR = '#c8d8e8'   # cool-white text

# Diverging colormap for difference.
# Midpoint (zero diff) is a neutral cool-grey so the glow "fades" to near-black
# when there is no difference, and burns blue or red for large differences.
CMAP_DIFF = mcolors.LinearSegmentedColormap.from_list(
    'diff_dark', ['#1565c0', '#d0d8e0', '#c62828'], N=512)

# Profile chart colours (publication palette)
CTRL_COLOR  = '#00e676'   # vivid green — control scenario
TEMPO_COLOR = '#ff8c00'   # amber — tempo-shifted scenario
FILL_POS    = '#ff4444'
FILL_NEG    = '#4444ff'

# 4-layer glow: (linewidth_multiplier, base_alpha)
GLOW_LAYERS = [(10.0, 0.018), (5.0, 0.070), (2.5, 0.180), (1.0, 0.850)]

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor':   BG_COLOR,
    'text.color':       TEXT_COLOR,
    'axes.labelcolor':  TEXT_COLOR,
    'xtick.color':      TEXT_COLOR,
    'ytick.color':      TEXT_COLOR,
    'axes.edgecolor':   DIM_COLOR,
    'font.family':      'monospace',
    'font.size':        9,
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
    """Return nested dict  loads[date][hour] = {region: total_kWh}"""
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


# ─── Step 5: Pre-compute geometry ────────────────────────────────────────────
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


def map_extent(nodes, pad_frac=0.04):
    xs = nodes.geometry.x
    ys = nodes.geometry.y
    dx, dy = xs.max() - xs.min(), ys.max() - ys.min()
    return (xs.min() - dx*pad_frac, xs.max() + dx*pad_frac,
            ys.min() - dy*pad_frac, ys.max() + dy*pad_frac)


# ─── Rendering primitives ─────────────────────────────────────────────────────
def draw_glow_edges(ax, segs, color, pulse=1.0, base_lw=0.7):
    """
    Draw network edges with a 4-layer glow.
    pulse ∈ [0, 1] scales glow alpha — higher = brighter/more vivid.
    """
    if not segs:
        return
    clamped = max(0.05, min(1.0, pulse))
    for width_mult, alpha in GLOW_LAYERS:
        lc = LineCollection(segs,
                            colors=[color],
                            linewidths=base_lw * width_mult,
                            alpha=alpha * clamped,
                            capstyle='round', joinstyle='round',
                            zorder=3)
        ax.add_collection(lc)


def draw_substation(ax, x, y, color, pulse=1.0):
    """
    Large, fixed-size landmark substation marker.
    • Three coloured halo rings that pulse with load difference.
    • Fixed-size white outer ring (always clearly visible as a landmark).
    • Fixed-size coloured inner fill.
    """
    clamped = max(0.10, min(1.0, pulse))
    # Coloured glow halos — pulse in intensity
    for size, alpha in [(800, 0.022), (450, 0.065), (240, 0.16)]:
        ax.scatter(x, y, s=size, color=color, alpha=alpha * clamped,
                   zorder=7, linewidths=0)
    # White outer ring — fixed, always visible
    ax.scatter(x, y, s=160, color='white', alpha=0.88, zorder=8, linewidths=0)
    # Coloured fill — fixed size
    ax.scatter(x, y, s=100, color=color, alpha=1.0, zorder=9, linewidths=0)
    # Tiny white centre pinpoint for precision
    ax.scatter(x, y, s=12,  color='white', alpha=0.95, zorder=10, linewidths=0)


def setup_map_ax(ax, xlim, ylim):
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect('equal')
    ax.axis('off')
    # Subtle geographic graticule (lat/lon grid)
    for lon in np.arange(-76.2, -75.4, 0.1):
        ax.axvline(lon, color=DIM_COLOR, lw=0.30, alpha=0.55, zorder=1)
    for lat in np.arange(37.0, 37.8, 0.1):
        ax.axhline(lat, color=DIM_COLOR, lw=0.30, alpha=0.55, zorder=1)


def fig_to_rgb(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    if h % 2: arr = arr[:-1, :, :]
    if w % 2: arr = arr[:, :-1, :]
    return arr


# ─── Output 1: 168-frame animated difference map ──────────────────────────────
def make_diff_animation(seg_map, substations, loads_ctrl, loads_tempo,
                        xlim, ylim):
    """
    7 days × 24 hours = 168 frames.

    Each substation's entire tree (edges + node) glows in a colour
    derived from the Tempo−Control load difference at that hour.
    Glow intensity (pulse) scales with |difference| so large deviations
    burn bright while near-zero differences fade toward the dark background.

    Substation dot is a FIXED size — only colour and halo brightness change.
    """
    print('\n[1/3] Building 168-frame difference animation…')

    regions = sorted(seg_map.keys())

    # Fixed colour scale across all 168 frames
    all_diffs = [
        loads_tempo[d][h].get(r, 0) - loads_ctrl[d][h].get(r, 0)
        for d in DAYS for h in range(1, 25) for r in regions
    ]
    abs_max = max(abs(v) for v in all_diffs) or 1.0
    norm    = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    print(f'  Global |Δload| range: ±{abs_max:.1f} kWh across all 168 frames')

    # Figure: single map panel + narrow colorbar
    fig = plt.figure(figsize=(14, 9), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            width_ratios=[1, 0.03],
                            left=0.01, right=0.94,
                            top=0.91, bottom=0.04,
                            wspace=0.025)
    ax  = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    # Colourbar (static — drawn once)
    cb = ColorbarBase(cax, cmap=CMAP_DIFF, norm=norm, orientation='vertical')
    cb.set_label('Δ load: Tempo − Control (kWh)', color=TEXT_COLOR, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cax.set_facecolor(BG_COLOR)
    cax.text(0.5,  1.02, 'Tempo\nincreases\nload', transform=cax.transAxes,
             ha='center', va='bottom', fontsize=7, color='#e57373',
             linespacing=1.3)
    cax.text(0.5, -0.02, 'Tempo\nreduces\nload', transform=cax.transAxes,
             ha='center', va='top', fontsize=7, color='#64b5f6',
             linespacing=1.3)

    fig.text(0.03, 0.012,
             'Glow intensity ∝ |Δload|.  '
             'White ring = substation landmark (fixed size).  '
             'Blue = tempo reduces load.  Red = tempo increases load.',
             color=DIM_COLOR, fontsize=7.5, va='bottom')

    # Pre-index substations for fast lookup
    subs_idx = substations.set_index('region')

    # Title Text objects — created once, updated in-place each frame
    sup_txt = fig.suptitle('', fontsize=12, fontweight='bold',
                           color='white', y=0.965)
    ttl_txt = ax.set_title('', fontsize=10, pad=5, loc='left')

    frames  = []
    frame_n = 0

    for date in DAYS:
        day_type  = DAY_TYPE[date]
        day_str   = {'red':   'Red Day  ·  tempo-shifting active',
                     'white': 'White Day  ·  baseline',
                     'blue':  'Blue Day  ·  baseline'}[day_type]
        badge_clr = {'red': '#ef5350', 'white': '#b0bec5',
                     'blue': '#42a5f5'}[day_type]

        for hr in range(1, 25):
            lc_data = loads_ctrl [date][hr]
            lt_data = loads_tempo[date][hr]

            ax.cla()
            setup_map_ax(ax, xlim, ylim)

            # ── Per-substation: coloured glow edges + landmark dot ────────────
            for region in regions:
                diff  = lt_data.get(region, 0) - lc_data.get(region, 0)
                color = CMAP_DIFF(norm(diff))
                # Pulse: minimum 0.12 so topology always faintly visible;
                # maximum 1.0 for the largest differences.
                pulse = 0.12 + 0.88 * abs(diff) / abs_max

                draw_glow_edges(ax, seg_map[region], color, pulse=pulse)

                row = subs_idx.loc[region]
                draw_substation(ax, row['x'], row['y'], color, pulse=pulse)

            # ── Update labels ────────────────────────────────────────────────
            sup_txt.set_text(
                f'{DAY_LABEL[date]}   ·   Hour  {hr:02d}:00')
            ttl_txt.set_text(day_str)
            ttl_txt.set_color(badge_clr)

            frames.append(fig_to_rgb(fig))
            frame_n += 1
            if hr == 1 or hr % 6 == 0:
                print(f'  {date}  hour {hr:02d}  [{frame_n}/168]', end='\r')

    plt.close(fig)

    out = os.path.join(OUTPUT_DIR, '01_diff_animation.mp4')
    imageio.mimwrite(out, frames, fps=4, quality=9, macro_block_size=1)
    print(f'\n  Saved → {out}  ({len(frames)} frames, {os.path.getsize(out)//1024} KB)')


# ─── Output 2: Substation 24h profiles — Red Day small multiples ─────────────
def make_substation_profiles(substations, loads_ctrl, loads_tempo):
    print('\n[2/3] Rendering substation profiles (Red Day small multiples)…')

    regions = sorted(substations['region'].tolist())
    n       = len(regions)
    cols    = 4
    rows    = (n + cols - 1) // cols
    hours   = list(range(1, 25))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4.5, rows * 3.2),
                             facecolor=BG_COLOR)
    axes_flat = axes.flatten()

    all_vals = []
    for r in regions:
        all_vals += [loads_ctrl[RED_DAY][h].get(r, 0) for h in hours]
        all_vals += [loads_tempo[RED_DAY][h].get(r, 0) for h in hours]
    ymax = max(all_vals) * 1.08

    for i, region in enumerate(regions):
        ax = axes_flat[i]
        ax.set_facecolor('#0c1622')
        ax.spines[:].set_color(DIM_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=6.5)

        ctrl_vals  = [loads_ctrl [RED_DAY][h].get(region, 0) for h in hours]
        tempo_vals = [loads_tempo[RED_DAY][h].get(region, 0) for h in hours]

        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=[t > c for t, c in zip(tempo_vals, ctrl_vals)],
                        color=FILL_POS, alpha=0.18, zorder=2)
        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=[t < c for t, c in zip(tempo_vals, ctrl_vals)],
                        color=FILL_NEG, alpha=0.18, zorder=2)
        ax.plot(hours, ctrl_vals,  color=CTRL_COLOR,  lw=1.6,
                solid_capstyle='round', zorder=3, label='Control')
        ax.plot(hours, tempo_vals, color=TEMPO_COLOR, lw=1.6,
                linestyle='--', dashes=(5, 2.5),
                solid_capstyle='round', zorder=4, label='Tempo-shifted')

        ax.set_title(f'Substation {region}', color=TEXT_COLOR,
                     fontsize=8.5, fontweight='bold', pad=3)
        ax.set_xlim(1, 24)
        ax.set_ylim(0, ymax)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xticklabels(['01', '06', '12', '18', '24'], fontsize=7)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.set_xlabel('Hour', color=TEXT_COLOR, fontsize=7, labelpad=2)
        ax.set_ylabel('Load (kWh)', color=TEXT_COLOR, fontsize=7, labelpad=2)
        ax.grid(color=DIM_COLOR, lw=0.4, alpha=0.7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    handles = [
        Line2D([0], [0], color=CTRL_COLOR,  lw=2,   label='Control'),
        Line2D([0], [0], color=TEMPO_COLOR, lw=2,
               linestyle='--', dashes=(5, 2.5), label='Tempo-shifted'),
        mpatches.Patch(color=FILL_POS, alpha=0.5,
                       label='Tempo > Control (load increase)'),
        mpatches.Patch(color=FILL_NEG, alpha=0.5,
                       label='Tempo < Control (load reduction)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               facecolor='#0c1622', edgecolor=DIM_COLOR,
               labelcolor=TEXT_COLOR,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f'24-Hour Substation Load Profiles  ·  Red Day ({RED_DAY})\n'
        'The Red Day is the only day on which the tempo-shifting tariff was active.',
        color='white', fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = os.path.join(OUTPUT_DIR, '02_substation_profiles.png')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  Saved → {out}')


# ─── Output 3: System-aggregate profiles — Red / White / Blue-average ─────────
def make_system_profiles(loads_ctrl, loads_tempo):
    print('\n[3/3] Rendering system-aggregate profiles (3 panels)…')

    hours = list(range(1, 25))

    def system_series(loads, date_or_dates):
        dates = [date_or_dates] if isinstance(date_or_dates, str) else list(date_or_dates)
        arr = np.array([
            [sum(loads[d][h].values()) for h in hours]
            for d in dates
        ])
        return arr.mean(axis=0)

    panels = [
        {'label': f'Red Day  ({RED_DAY})',
         'sublabel': 'Tempo-shifting active',
         'dates': RED_DAY, 'badge': '#ef5350'},
        {'label': f'White Day  ({WHITE_DAY})',
         'sublabel': 'Baseline (no tempo-shifting)',
         'dates': WHITE_DAY, 'badge': '#b0bec5'},
        {'label': 'Blue Days  (average)',
         'sublabel': f'Mean of {len(BLUE_DAYS)} baseline days',
         'dates': BLUE_DAYS, 'badge': '#42a5f5'},
    ]

    all_vals = []
    for p in panels:
        all_vals += list(system_series(loads_ctrl,  p['dates']))
        all_vals += list(system_series(loads_tempo, p['dates']))
    ymax = max(all_vals) * 1.08

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=BG_COLOR,
                             sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.18,
                        wspace=0.06)

    for ax, p in zip(axes, panels):
        ctrl_vals  = system_series(loads_ctrl,  p['dates'])
        tempo_vals = system_series(loads_tempo, p['dates'])

        ax.set_facecolor('#0c1622')
        ax.spines[:].set_color(DIM_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=tempo_vals > ctrl_vals,
                        color=FILL_POS, alpha=0.18, zorder=1)
        ax.fill_between(hours, ctrl_vals, tempo_vals,
                        where=tempo_vals < ctrl_vals,
                        color=FILL_NEG, alpha=0.18, zorder=1)
        ax.plot(hours, ctrl_vals,  color=CTRL_COLOR,  lw=2.2,
                solid_capstyle='round', zorder=3, label='Control')
        ax.plot(hours, tempo_vals, color=TEMPO_COLOR, lw=2.2,
                linestyle='--', dashes=(6, 3),
                solid_capstyle='round', zorder=4, label='Tempo-shifted')

        if 'Red' in p['label']:
            peak_h = int(np.argmax(ctrl_vals)) + 1
            peak_v = ctrl_vals[peak_h - 1]
            ax.annotate(f'Peak\n{peak_v/1e6:.2f} GWh',
                        xy=(peak_h, peak_v),
                        xytext=(peak_h - 4, peak_v * 0.92),
                        arrowprops=dict(arrowstyle='->', color=TEXT_COLOR, lw=0.8),
                        fontsize=8, color=TEXT_COLOR)

        ax.set_xlim(1, 24)
        ax.set_ylim(0, ymax)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xticklabels(['01:00', '06:00', '12:00', '18:00', '24:00'],
                           fontsize=8.5, color=TEXT_COLOR)
        ax.set_xlabel('Hour of day', color=TEXT_COLOR, fontsize=9.5, labelpad=4)
        ax.grid(color=DIM_COLOR, lw=0.5, alpha=0.7)
        ax.set_title(p['label'], fontsize=12, fontweight='bold',
                     color=p['badge'], pad=6)
        ax.text(0.5, 1.035, p['sublabel'], transform=ax.transAxes,
                ha='center', fontsize=9, color=TEXT_COLOR, style='italic')

    axes[0].set_ylabel('Total system load (kWh)', color=TEXT_COLOR,
                       fontsize=10, labelpad=6)
    axes[0].yaxis.set_tick_params(labelsize=9, colors=TEXT_COLOR)

    handles = [
        Line2D([0], [0], color=CTRL_COLOR,  lw=2.2, label='Control'),
        Line2D([0], [0], color=TEMPO_COLOR, lw=2.2,
               linestyle='--', dashes=(6, 3), label='Tempo-shifted'),
        mpatches.Patch(color=FILL_POS, alpha=0.5,
                       label='Tempo > Control'),
        mpatches.Patch(color=FILL_NEG, alpha=0.5,
                       label='Tempo < Control'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9.5,
               facecolor='#0c1622', edgecolor=DIM_COLOR,
               labelcolor=TEXT_COLOR, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        'System-Aggregate 24-Hour Load Profile  ·  Control vs. Tempo-Shifted\n'
        '21 substations  ·  28,195 households  ·  July 2014  ·  '
        'Blue-day panel = mean of 5 baseline days',
        color='white', fontsize=12, fontweight='bold', y=0.98)

    out = os.path.join(OUTPUT_DIR, '03_system_profiles.png')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
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

    print('\nPre-computing edge segments…')
    seg_map     = build_edge_segments(edges)
    substations = extract_substations(nodes)

    x0, x1, y0, y1 = map_extent(nodes)
    xlim, ylim = (x0, x1), (y0, y1)
    print(f'  Extent: lon [{x0:.3f}, {x1:.3f}]  lat [{y0:.3f}, {y1:.3f}]')

    make_diff_animation(seg_map, substations, loads_ctrl, loads_tempo,
                        xlim, ylim)

    make_substation_profiles(substations, loads_ctrl, loads_tempo)

    make_system_profiles(loads_ctrl, loads_tempo)

    print('\n=== Done.  Outputs in:', OUTPUT_DIR, '===')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        print(f'  {f:45s}  {os.path.getsize(path)//1024:>6} KB')


if __name__ == '__main__':
    main()
