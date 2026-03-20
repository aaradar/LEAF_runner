import os
import sys
from pathlib import Path

# --------------------------------------------------
# FIX 1: Always set paths relative to THIS script
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# Add project root to python path
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "source"))

print("Running from:", SCRIPT_DIR)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
from prepare_params import prepare_production_params
from Production import main

# --------------------------------------------------
# FILE PATHS (USE RAW STRINGS — VERY IMPORTANT ON WINDOWS)
# --------------------------------------------------
kml = r"Sample Points\AfforestationSItesFixed.kml"
shp = r"Sample Points\FieldPoints32_2018.shp"
tmx = r"Sample Points\TMX\TML_pipeline_100mbuffer.kml"
cbc = r"Sample Points\ColdwaterBCregion.kml"

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
ProdParams = {
    'sensor': 'S2_SR',
    'unit': 2,
    'nbYears': -1,

    'regions': kml,
    'file_variables': ['TARGET_FID', 'AsssD_1', 'AsssD_2'],
    'regions_start_index': 0,
    'regions_end_index': 5,

    'temporal_buffer': [["2025-08-01", "2025-08-31"]],

    'resolution': 10,
    'projection': 'EPSG:3979',
    'prod_names': ['mosaic'],
    'out_folder': r'E:\S2_mosaics_runner_2026\Afforestation\S2_py_10m',
    'out_datatype': 'int16'
}

CompParams = {
    "number_workers": 32,
    "debug": True,
    "entire_tile": False,
    "nodes": 1,
    "node_memory": "16G",
    'chunk_size': {'x': 512, 'y': 512}
}

# --------------------------------------------------
# RUN PIPELINE
# --------------------------------------------------
print("Preparing parameters...")
result = prepare_production_params(ProdParams, CompParams)

print("Starting production...")
main(ProdParams, CompParams)

print("Finished successfully.")