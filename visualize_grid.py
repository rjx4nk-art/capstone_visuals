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


# Map background color — warm cream, echoes OSM Positron palette
MAP_BG      = '#f5f1e8'
# Phantom "street" color — all edges drawn once as a ghost layer
STREET_CLR  = '#ccc9be'


def map_xlim_ylim(nodes_3857, pad_frac=0.05):
    xs = nodes_3857.geometry.x
    ys = nodes_3857.geometry.y
    dx, dy = xs.max() - xs.min(), ys.max() - ys.min()
    return ((xs.min() - dx*pad_frac, xs.max() + dx*pad_frac),
            (ys.min() - dy*pad_frac, ys.max() + dy*pad_frac))


def setup_map_ax(ax, xlim, ylim):
    """Configure a clean map axis with warm-cream background."""
    ax.set_facecolor(MAP_BG)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')


def fig_to_rgb(fig):
    """Convert a matplotlib figure to an H×W×3 uint8 array (even dimensions)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=MAP_BG)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    if h % 2: arr = arr[:-1, :, :]
    if w % 2: arr = arr[:, :-1, :]
    return arr


# ─── Output 1: Animated difference map (7 days × 24 hours = 168 frames) ───────
def make_diff_animation(seg_map_3857, subs_3857, loads_ctrl, loads_tempo,
                        xlim, ylim):
    """
    Each frame:
      • Warm-cream background  (mimics OSM Positron palette)
      • Ghost street layer     — entire network in light warm gray (lw=0.35)
                                 Distribution lines follow streets → reads as a
                                 street map with no external tile download needed.
      • Per-substation glow    — 3-layer LineCollection (wide glow / mid glow /
                                 core line) colored by Tempo−Control difference.
                                 Glow intensity (alpha) scales with |diff|.
      • Fixed-size substation  — constant s=240, colored by diff, white halo.
    """
    print('\n[1/3] Building difference animation (168 frames)…')

    regions = sorted(seg_map_3857.keys())

    # ── Global diff range (fixed colorbar across all 168 frames) ────────────
    all_diffs = np.array([
        loads_tempo[d][h].get(r, 0) - loads_ctrl[d][h].get(r, 0)
        for d in DAYS for h in range(1, 25) for r in regions
    ])
    abs_max = float(np.abs(all_diffs).max()) or 1.0
    norm    = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # ── Pre-compute segment list and which region each segment belongs to ────
    all_segs    = []
    seg_reg_idx = []   # index into `regions` list
    for ri, r in enumerate(regions):
        for seg in seg_map_3857[r]:
            all_segs.append(seg)
            seg_reg_idx.append(ri)
    seg_reg_idx = np.array(seg_reg_idx, dtype=int)   # shape: (N_segs,)
    n_segs      = len(all_segs)
    print(f'  {n_segs:,} edge segments across {len(regions)} substations')

    # ── Substation coordinates (fixed) ──────────────────────────────────────
    subs_idx = subs_3857.set_index('region')
    xs_s = np.array([subs_idx.loc[r, 'x'] for r in regions])
    ys_s = np.array([subs_idx.loc[r, 'y'] for r in regions])

    # ── Figure layout: wide map | narrow colorbar ────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor=MAP_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            width_ratios=[1, 0.028],
                            left=0.01, right=0.94,
                            top=0.90, bottom=0.04,
                            wspace=0.025)
    ax  = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    cb = ColorbarBase(cax, cmap=CMAP_DIFF, norm=norm, orientation='vertical')
    cb.set_label('Tempo − Control  (kWh)', fontsize=9, labelpad=6)
    cb.ax.yaxis.set_tick_params(labelsize=8)
    cb.ax.set_facecolor(MAP_BG)
    cax.text(0.5,  1.025, 'Load\nincrease', transform=cax.transAxes,
             ha='center', va='bottom', fontsize=7, color='#b2182b',
             linespacing=1.3)
    cax.text(0.5, -0.025, 'Load\nreduction', transform=cax.transAxes,
             ha='center', va='top',    fontsize=7, color='#2166ac',
             linespacing=1.3)

    fig.text(0.03, 0.012,
             'Distribution network overlaid on geographic extent.  '
             'Substation dot color = Tempo − Control difference.  '
             'Glow intensity ∝ |difference|.',
             fontsize=7.5, color='#777', va='bottom')

    # ── Ghost "street" layer — drawn ONCE, never cleared ─────────────────────
    # This is a static LineCollection added directly to the axes at z=1.
    # It stays across ax.cla() because we only clear and redraw the dynamic
    # elements (glow layers + substation dots + title) manually below.
    # We handle this by NOT calling ax.cla() — instead we remove and re-add
    # only the dynamic artists.
    setup_map_ax(ax, xlim, ylim)
    ghost_lc = LineCollection(all_segs,
                              colors=STREET_CLR, linewidths=0.35, alpha=0.80,
                              capstyle='round', zorder=1)
    ax.add_collection(ghost_lc)

    # Create title Text objects ONCE; update text/color each frame in-place
    sup_txt = fig.suptitle('', fontsize=13, fontweight='bold',
                           color='#111111', y=0.965)
    ttl_txt = ax.set_title('', fontsize=10, pad=5, loc='left')

    # Removable dynamic artists (LineCollections + scatter)
    dynamic_artists = []

    frames  = []
    frame_n = 0

    for date in DAYS:
        day_type  = DAY_TYPE[date]
        day_str   = {'red':   'Red Day  —  tempo-shifting active',
                     'white': 'White Day  —  baseline',
                     'blue':  'Blue Day  —  baseline'}[day_type]
        badge_clr = {'red': '#c0392b', 'white': '#555555',
                     'blue': '#1a6fad'}[day_type]

        for hr in range(1, 25):
            lc_data = loads_ctrl [date][hr]
            lt_data = loads_tempo[date][hr]

            # ── Remove previous dynamic artists ─────────────────────────────
            for art in dynamic_artists:
                art.remove()
            dynamic_artists.clear()

            # ── Per-segment colors  (vectorized) ────────────────────────────
            diff_by_region = np.array(
                [lt_data.get(r, 0) - lc_data.get(r, 0) for r in regions]
            )                                                   # shape: (21,)
            diff_per_seg   = diff_by_region[seg_reg_idx]       # shape: (N,)
            rgba           = CMAP_DIFF(norm(diff_per_seg))     # (N, 4)

            # pulse ∈ [0.25, 1.0] — drives glow intensity
            pulse = 0.25 + 0.75 * np.abs(diff_per_seg) / abs_max

            # Wide outer glow  — soft halo
            wide_c       = rgba.copy(); wide_c[:, 3] = pulse * 0.10
            lc_wide = LineCollection(all_segs, colors=wide_c,
                                     linewidths=18, capstyle='round', zorder=2)
            # Mid glow
            mid_c        = rgba.copy(); mid_c[:, 3]  = pulse * 0.25
            lc_mid  = LineCollection(all_segs, colors=mid_c,
                                     linewidths=6,  capstyle='round', zorder=3)
            # Core line
            core_c       = rgba.copy()
            core_c[:, 3] = np.clip(pulse * 0.95, 0.55, 0.95)
            lc_core = LineCollection(all_segs, colors=core_c,
                                     linewidths=1.4, capstyle='round', zorder=4)

            ax.add_collection(lc_wide)
            ax.add_collection(lc_mid)
            ax.add_collection(lc_core)
            dynamic_artists += [lc_wide, lc_mid, lc_core]

            # ── Substation dots: FIXED size, colored by diff ─────────────────
            sub_colors = CMAP_DIFF(norm(diff_by_region))   # (21, 4)

            halo = ax.scatter(xs_s, ys_s, s=390, c='white',
                              zorder=6, linewidths=0)
            dot  = ax.scatter(xs_s, ys_s, s=260, c=sub_colors,
                              zorder=7, edgecolors='#222222', linewidths=0.9)
            dynamic_artists += [halo, dot]

            # ── Update labels in-place ────────────────────────────────────────
            sup_txt.set_text(f'{DAY_LABEL[date]}   |   Hour  {hr:02d}:00')
            ttl_txt.set_text(day_str)
            ttl_txt.set_color(badge_clr)

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

    # Reproject to Web Mercator (EPSG:3857) so geometry is in metres
    print('\nReprojecting network to EPSG:3857…')
    nodes_3857 = nodes.to_crs('EPSG:3857')
    edges_3857 = edges.to_crs('EPSG:3857')

    seg_map_3857 = build_edge_segments(edges_3857)
    subs_3857    = extract_substations(nodes_3857)
    xlim, ylim   = map_xlim_ylim(nodes_3857)

    print(f'  Map extent (EPSG:3857):  '
          f'x [{xlim[0]:.0f}, {xlim[1]:.0f}]  '
          f'y [{ylim[0]:.0f}, {ylim[1]:.0f}]')
    print('  Background: phantom street layer from distribution network geometry')

    # ── Generate outputs ─────────────────────────────────────────────────────
    make_diff_animation(seg_map_3857, subs_3857,
                        loads_ctrl, loads_tempo, xlim, ylim)

    make_substation_profiles(subs_3857, loads_ctrl, loads_tempo)

    make_system_profiles(loads_ctrl, loads_tempo)

    print('\n=== Done.  Outputs in:', OUTPUT_DIR, '===')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        print(f'  {f:45s}  {os.path.getsize(path)//1024:>6} KB')


if __name__ == '__main__':
    main()
