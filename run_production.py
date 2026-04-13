import os
import sys
from pathlib import Path

# --------------------------------------------------
# PATH SETUP  (mirrors notebook cell 0)
# These must stay at module level so that worker
# processes spawned by Dask/multiprocessing can
# import this file without re-running the pipeline.
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "source"))

from prepare_params import prepare_production_params
from Production import main
import geopandas as gpd  # noqa: F401  (kept to match notebook cell 4 imports)

# --------------------------------------------------
# FILE PATHS  (mirrors notebook cell 0)
# --------------------------------------------------
kml     = r"Sample Points\AfforestationSItesFixed.kml"
shp     = r"Sample Points\FieldPoints32_2018.shp"
tmx     = r"Sample Points\TMX\TML_pipeline_100mbuffer.kml"
cbc     = r"Sample Points\ColdwaterBCregion.kml"
ls_mgrs = "S2A_OPER_GIP_TILPAR_MPC.kml"
s2_mgrs = "sentinel-2-grid.parquet"

# --------------------------------------------------
# ALL EXECUTION MUST BE UNDER THIS GUARD.
# On Windows, multiprocessing uses "spawn" instead
# of "fork", which means every worker re-imports
# this file as __main__. Without the guard, each
# worker immediately tries to launch more workers,
# causing the bootstrapping crash seen in the logs.
# --------------------------------------------------
if __name__ == '__main__':

    print("Modules imported successfully")
    print(f"Current working directory: {os.getcwd()}")
    print(kml)
    print(shp)
    print(tmx)
    print(cbc)

    # Use Case: Generate monthly mosaics
    # Customize these parameters as needed:

    ProdParams = {
        # ============ REQUIRED PARAMETERS ============
        'sensor': 'S2_SR',          # A sensor type string (e.g., 'S2_SR', 'HLS_SR', 'HLSL30_SR', 'HLSS30_SR' or 'MOD_SR')
        'unit': 2,                   # A data unit code (1 or 2 for TOA or surface reflectance)
        'nbYears': -1,               # positive int for annual product, or negative int for monthly product
        #'year': 2023,               # CHANGE THIS: Year to process
        #'months': [8, 9, 10],       # CHANGE THIS: Months to process (1-12)

        # ============ REGION PARAMETERS ============
        'regions': kml,              # CHANGE THIS: Path to your KML/SHP file
        's2_grid_path': s2_mgrs,
        'mode': 'regions',           # regions or tiles
        'subdivide_tiles': False,    # Whether to subdivide tiles into smaller regions (only applicable if mode is 'tiles')
        'file_variables': ['TARGET_FID', 'AsssD_1', 'AsssD_2'],  # CHANGE THIS: id, start_date, end_date (None for dates if they don't exist)
        #'file_variables': ['id', 'begin', 'end'],
        #'file_variables': ['system:index', None, None],
        'regions_start_index': 3000, # CHANGE THIS: Start at this region index
        'regions_end_index': 3002,   # CHANGE THIS: End at this index (None = all)

        # ============ BUFFER PARAMETERS ============
        #'spatial_buffer_m': -10,                          # UNCOMMENT & CHANGE: Buffer in meters around regions
        #'temporal_buffer': [[50, 90], [-10, 10]],         # UNCOMMENT & CHANGE: [days_before, days_after]
        'temporal_buffer': [["2025-08-01", "2025-08-31"]],
        #'num_years': 10,

        # ============ OUTPUT PARAMETERS ============
        'resolution': 30,            # Resolution in meters
        'projection': 'EPSG:3979',   # Coordinate projection
        'IncludeAngles': False,
        'prod_names': ['mosaic'],       # ['mosaic', 'LAI', 'fCOVER', 'fAPAR', 'Albedo']
        'out_folder': r'E:\Testing1\hls_lai2_10m',
        'out_datatype': 'int16',

        # ============ CSV OUTPUT ============
        'output_type': 'geotiff',    # switch to 'csv' to export as CSV instead of GeoTIFF
        'csv_scale': 10,             # optional: same as resolution, explicit here for clarity
        'csv_dropNulls': True,       # optional: skip masked/nodata pixels (mirrors GEE dropNulls)
        'csv_max_pixels': 100_000,   # optional: lower this first for a quick test
    }

    CompParams = {
        "number_workers": 10,
        "debug": True,
        "entire_tile": False,
        "nodes": 1,
        "node_memory": "50G",
        'chunk_size': {'x': 512, 'y': 512}
    }

    print("\nconfigured")

    # STEP 1 — prepare parameters  (mirrors notebook cell 2)
    # prepare_production_params mutates ProdParams in-place,
    # populating region_start_dates, region_end_dates, and
    # converting the regions path into a dict of GeoJSON polygons.
    result = prepare_production_params(ProdParams, CompParams)

    # STEP 2 — run production  (mirrors notebook cells 4-5)
    print(ProdParams)
    main(ProdParams, CompParams)