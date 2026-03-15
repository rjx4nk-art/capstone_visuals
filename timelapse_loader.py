import os
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────
NODELIST_SHP = r"C:\Users\danny\Downloads\dist_net\content\output\merged\merged_nodelist.shp"
CSV_PATH     = r"C:\Users\danny\Downloads\dist_net\content\output\hid_loads_toy.csv"
OUTPUT_DIR   = r"C:\Users\danny\Downloads\dist_net\content\output\merged"
OUTPUT_NAME  = "nodelist_timelapse.gpkg"
START_DATE   = datetime(2024, 1, 1, 0, 0, 0)
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading shapefile...")
nodes = gpd.read_file(NODELIST_SHP)

print("Loading CSV...")
csv_df = pd.read_csv(CSV_PATH, dtype={"hid": str})

# ── Normalize HID to string on both sides ─────────────────────────────────────
print("\nNormalizing HID columns...")

# Shapefile side — drop the .0 from floats, skip nulls
nodes["hid_key"] = nodes["hid"].apply(
    lambda x: str(int(float(x))) if pd.notna(x) else None
)

# CSV side — straightforward int → string
csv_df["hid_key"] = csv_df["hid"].apply(
    lambda x: str(int(float(x))) if pd.notna(x) else None
)

# Show only non-null samples so we can actually see the values
shp_sample = nodes["hid_key"].dropna().head(5).tolist()
csv_sample = csv_df["hid_key"].dropna().head(5).tolist()
print(f"  Shapefile non-null HID sample: {shp_sample}")
print(f"  CSV non-null HID sample:       {csv_sample}")

# ── Time step columns ──────────────────────────────────────────────────────────
time_cols = [c for c in csv_df.columns if c not in ("hid", "hid_key")]
print(f"\n  Found {len(time_cols)} time-step columns: {time_cols}")

# ── Join on hid_key ────────────────────────────────────────────────────────────
print("\nJoining CSV to shapefile on normalized HID...")
nodes_merged = nodes.merge(csv_df[["hid_key"] + time_cols], on="hid_key", how="left")

unmatched = nodes_merged[time_cols[0]].isna().sum()
total     = len(nodes_merged)
print(f"  Matched:   {total - unmatched} / {total} nodes")
if unmatched > 0:
    print(f"  Unmatched: {unmatched} nodes")

# ── Reshape to long format ─────────────────────────────────────────────────────
print("\nReshaping to long format...")
timestamps = [START_DATE + timedelta(hours=i) for i in range(len(time_cols))]

long_frames = []
for i, col in enumerate(time_cols):
    frame = nodes_merged[["hid_key", col, "geometry"]].copy()
    frame = frame.rename(columns={col: "value"})
    frame["timestamp"] = timestamps[i]
    frame["hour"]      = i
    long_frames.append(frame)

long_gdf = pd.concat(long_frames, ignore_index=True)
long_gdf = gpd.GeoDataFrame(long_gdf, geometry="geometry", crs=nodes.crs)
long_gdf["value"] = pd.to_numeric(long_gdf["value"], errors="coerce").astype("Int64")

print(f"  Long-format GDF: {len(long_gdf)} rows ({total} nodes × {len(time_cols)} hours)")

# ── Write GeoPackage ───────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
print(f"\nWriting output to: {out_path}")
long_gdf.to_file(out_path, driver="GPKG")
print("Done.")

print("""
─────────────────────────────────────────────────────
Next steps in ArcGIS Pro:
  1. Add Data → select nodelist_timelapse.gpkg
  2. Right-click layer → Properties → Time tab
  3. Time Field: timestamp
  4. Click OK → Time Slider activates
  5. Symbolize 'value' with a graduated colour ramp
  6. Hit Play
─────────────────────────────────────────────────────
""")