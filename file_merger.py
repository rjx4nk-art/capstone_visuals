import os
import glob
import geopandas as gpd

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = r"C:\Users\danny\Downloads\dist_net\content\output"
OUTPUT_DIR = r"C:\Users\danny\Downloads\dist_net\content\output\merged"
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

edge_gdfs = []
node_gdfs = []

# Walk through every subfolder in the output directory
for folder_name in os.listdir(INPUT_DIR):
    folder_path = os.path.join(INPUT_DIR, folder_name)

    # Skip the output folder itself and any non-directories
    if not os.path.isdir(folder_path) or folder_name == "merged":
        continue

    # Find the edgelist and nodelist shapefiles for this region
    edge_pattern = os.path.join(folder_path, f"{folder_name}-edgelist.shp")
    node_pattern = os.path.join(folder_path, f"{folder_name}-nodelist-HID.shp")

    edge_files = glob.glob(edge_pattern)
    node_files = glob.glob(node_pattern)

    if edge_files:
        print(f"  Reading edges:  {edge_files[0]}")
        gdf = gpd.read_file(edge_files[0])
        gdf["source_region"] = folder_name   # optional: track origin
        edge_gdfs.append(gdf)
    else:
        print(f"  [WARN] No edgelist found in: {folder_path}")

    if node_files:
        print(f"  Reading nodes:  {node_files[0]}")
        gdf = gpd.read_file(node_files[0])
        gdf["source_region"] = folder_name   # optional: track origin
        node_gdfs.append(gdf)
    else:
        print(f"  [WARN] No nodelist found in: {folder_path}")

# ── Concatenate and write output ───────────────────────────────────────────────
if edge_gdfs:
    merged_edges = gpd.pd.concat(edge_gdfs, ignore_index=True)
    # Re-project all to a common CRS if needed (uses the first file's CRS)
    merged_edges = merged_edges.to_crs(edge_gdfs[0].crs)
    out_path = os.path.join(OUTPUT_DIR, "merged_edgelist.shp")
    merged_edges.to_file(out_path)
    print(f"\n✓ Merged edges saved  → {out_path}  ({len(merged_edges)} features)")
else:
    print("\n[ERROR] No edge shapefiles were found.")

if node_gdfs:
    merged_nodes = gpd.pd.concat(node_gdfs, ignore_index=True)
    merged_nodes = merged_nodes.to_crs(node_gdfs[0].crs)
    out_path = os.path.join(OUTPUT_DIR, "merged_nodelist.shp")
    merged_nodes.to_file(out_path)
    print(f"✓ Merged nodes saved  → {out_path}  ({len(merged_nodes)} features)")
else:
    print("\n[ERROR] No node shapefiles were found.")