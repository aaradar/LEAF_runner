# LEAF Runner — Parameter Preparation System

## Installation

**Option A: Conda environment (recommended)**
```bash
conda create -n leaf-env python=3.11.14
conda activate leaf-env

# Step 1: Core dependencies
conda install -c conda-forge click==8.1.7 dask==2024.5.2 dask-jobqueue==0.9.0 \
  numpy==1.24.4 odc-geo==0.4.8 odc-stac==0.3.10 pandas==2.2.3 psutil==5.9.8 \
  pyproj==3.6.1 pystac-client==0.8.2 rasterio==1.3.10 Requests==2.32.3 \
  rioxarray==0.15.6 stackstac==0.5.1 tqdm==4.66.4 urllib3==2.3.0 \
  xarray==2024.6.0 "bokeh!=3.0.*,>=2.4.2" gdal==3.9.2

# Step 2: Spatial libraries
conda install -c conda-forge geopandas==1.1.2 shapely==2.0.6 pyogrio==0.10.0 packaging==25.0

# Step 3 (optional): Jupyter kernel
conda install ipykernel==7.1.0
python -m ipykernel install --user --name leaf-env --display-name "Python (leaf-env)"
```

**Option B: From environment file**
```bash
conda env create -f environment.yml
conda activate leaf-env
python -m ipykernel install --user --name leaf-env --display-name "Python (leaf-env)"
```

---

## Running Production

**Terminal**
```bash
conda activate leaf-env
python run_production.py
```

**Jupyter**
```bash
jupyter notebook        # or jupyter lab
# Open parameter_preparation.ipynb
```

---

## Overview

The Parameter Preparation System handles pre-production validation, region loading, temporal window generation, and polygon filtering before calling `Production.py`.

### Key Files

| File | Purpose |
|------|---------|
| `prepare_params.py` | Main orchestrator — validates, loads regions, generates windows |
| `run_production.py` | Terminal entry point |
| `parameter_preparation.ipynb` | Interactive notebook with 4 templates |
| `source/leaf_wrapper.py` | Loads KML/SHP → LEAF region dicts |
| `source/s2_tile_processor.py` | Converts regions to S2 MGRS tile footprints (`mode='tiles'`) |
| `source/polygon_validator.py` | Filters zero-area polygons, creates CSV log |
| `Production.py` | Main production script (region-specific date support) |

---

## ProdParams Reference

### Required

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `sensor` | str | `'S2_SR'` | Sensor type. Options: `'S2_SR'`, `'HLS_SR'`, `'HLSL30_SR'`, `'HLSS30_SR'`, `'MOD_SR'` |
| `unit` | int | `2` | Data unit. `1` = TOA reflectance, `2` = surface reflectance |
| `nbYears` | int | `-1` | Positive int for annual product, negative int for monthly product |
| `regions` | str or dict | `'./sites.kml'` | Path to `.kml` / `.shp` file, or a pre-built region dict |
| `resolution` | int | `30` | Output resolution in metres |
| `projection` | str | `'EPSG:3979'` | Output coordinate reference system |
| `out_folder` | str | `r'E:\output\run1'` | Directory where outputs are written |

### Region / File Parameters

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `file_variables` | list | `['SiteID', 'AsssD_1', 'AsssD_2']` | Three elements: `[id_column, start_date_column, end_date_column]`. Date columns can be `None` if not in the file. ID column cannot be `None`. |
| `regions_start_index` | int | `0` | First region to load (0-based, inclusive). Default: `0` |
| `regions_end_index` | int\|None | `50` | Last region to load (0-based, inclusive). `None` = load all |
| `mode` | str | `'regions'` | `'regions'` (default) or `'tiles'` — see Processing Modes |
| `s2_grid_path` | str | `'sentinel-2-grid.parquet'` | Path to the S2 MGRS tile grid parquet file. Required when `mode='tiles'` |
| `subdivide_tiles` | bool | `False` | Subdivide S2 tile footprints into smaller sub-regions. Only applies when `mode='tiles'` |

### Temporal Parameters

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `year` | int | `2023` | Base year for month-based temporal modes |
| `months` | list[int] | `[6, 7, 8]` | Months to process (1–12). Used with `year` |
| `num_years` | int | `3` | Repeat `months` across this many years starting from `year` (e.g., `year=2020`, `num_years=3` → 2020, 2021, 2022) |
| `start_dates` | list[str] | `['2023-06-01']` | Custom window start dates in `YYYY-MM-DD` format |
| `end_dates` | list[str] | `['2023-06-30']` | Custom window end dates in `YYYY-MM-DD` format |
| `start_date` | str | `'2023-06-01'` | Single start date; end date is auto-copied to match |
| `end_date` | str | `'2023-06-30'` | Single end date; start date is auto-copied to match |
| `temporal_buffer` | list | `[[-7, 7]]` | Shift or replace date windows — see Temporal Buffer |

### Spatial Buffer

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `spatial_buffer_m` | int\|float | `-20` | Buffer applied to region geometries in metres before processing. **Negative** = erode (shrink inward); **positive** = dilate (expand outward). Reprojected to EPSG:3979 (Canada Atlas Lambert) for metric accuracy, then back to original CRS. If a negative buffer collapses a polygon to a point, it is flagged in the processing log and skipped. |

### Output Parameters

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `prod_names` | list[str] | `['mosaic']` | Products to generate. Options: `'mosaic'`, `'LAI'`, `'fCOVER'`, `'fAPAR'`, `'Albedo'` |
| `out_datatype` | str | `'int16'` | Output raster data type (e.g., `'int16'`, `'float32'`) |
| `IncludeAngles` | bool | `False` | Include solar/view angle bands in output |
| `output_type` | str | `'geotiff'` | Output format: `'geotiff'` or `'csv'` |
| `csv_scale` | int | `30` | Pixel size for CSV export (typically same as `resolution`) |
| `csv_dropNulls` | bool | `True` | Skip masked/nodata pixels in CSV output |
| `csv_max_pixels` | int | `100_000` | Max pixels exported per region in CSV mode. Reduce for quick test runs |

---

## CompParams Reference

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `number_workers` | int | `10` | Number of Dask workers to spawn |
| `debug` | bool | `True` | Enable verbose debug output |
| `entire_tile` | bool | `False` | Process the full S2 tile extent rather than clipping to the region polygon |
| `nodes` | int | `1` | Number of compute nodes (for HPC/cluster use) |
| `node_memory` | str | `'50G'` | Memory allocated per node (e.g., `'16G'`, `'50G'`) |
| `chunk_size` | dict | `{'x': 512, 'y': 512}` | Dask spatial chunk size in pixels. Smaller chunks use less memory; larger chunks reduce overhead |

---

## Processing Modes

### `mode='regions'` (default)
Each KML/SHP polygon is used directly as a LEAF processing region. Output is clipped to each polygon's extent.

### `mode='tiles'`
Finds all S2 MGRS tiles that intersect the input polygons and processes those full tile footprints instead. Useful for full-tile mosaics aligned to the S2 grid.

```python
'mode': 'tiles',
's2_grid_path': 'sentinel-2-grid.parquet',
'subdivide_tiles': False,   # True to further split tile footprints
```

In tiles mode, three extra keys are stored in `ProdParams` for reference after processing:

| Key | Contents |
|-----|---------|
| `regions_ref` | Original polygon geometries keyed by polygon ID |
| `region_start_dates_ref` | Start dates keyed by original polygon ID |
| `region_end_dates_ref` | End dates keyed by original polygon ID |

These mirror the live tile-keyed results and are useful for tracing which tiles came from which original polygon.

---

## Temporal Modes

`form_time_windows()` selects a mode based on which keys are present in `ProdParams`:

| Mode | Keys Required | Behaviour |
|------|--------------|-----------|
| 1 — Monthly | `year` + `months` | One window per month (1st → last day of month) |
| 2 — Custom ranges | `start_dates` + `end_dates` | Used as-is |
| 3 — Single date | `start_date` OR `end_date` | Missing end is auto-copied from the other |
| 4 — Multi-year monthly | `year` + `months` + `num_years` | Repeats each month across N consecutive years |

If none of these keys are found, temporal generation is skipped and region-specific dates from the file are used directly.

### Temporal Buffer

Applies on top of whichever mode above is active. Two sub-modes:

**Offset mode** — integers, shift each window's edges by N days:
```python
'temporal_buffer': [[-7, 7]]                      # widen each window by ±7 days
'temporal_buffer': [[-5, 5], [-10, 10], [0, 15]]  # each region date becomes 3 windows
```
When multiple pairs are provided, each original date pair expands into that many windows. Multiple pairs only apply to region-specific dates (from file); for global `year`/`months` windows, only a single pair is supported.

**Override mode** — date strings, replace all dates with fixed windows regardless of file dates:
```python
'temporal_buffer': [["2025-08-01", "2025-08-31"]]
'temporal_buffer': [["2024-04-15", "2024-07-15"], ["2024-08-01", "2024-09-01"]]
```

Both modes apply in parallel to live (tile-keyed) and ref (polygon-keyed) date dicts when `mode='tiles'`.

---

## Usage Examples

### Minimal — regions mode, terminal run

Edit `run_production.py` then:
```bash
conda activate leaf-env
python run_production.py
```

```python
ProdParams = {
    'sensor': 'S2_SR',
    'unit': 2,
    'nbYears': -1,
    'regions': r'Sample Points\AfforestationSItesFixed.kml',
    'file_variables': ['TARGET_FID', 'AsssD_1', 'AsssD_2'],
    'regions_start_index': 0,
    'regions_end_index': 10,
    'mode': 'regions',
    'temporal_buffer': [["2025-08-01", "2025-08-31"]],
    'resolution': 30,
    'projection': 'EPSG:3979',
    'prod_names': ['mosaic'],
    'out_folder': r'E:\output\run1',
    'out_datatype': 'int16',
    'output_type': 'geotiff',
    'IncludeAngles': False,
}
CompParams = {
    'number_workers': 10,
    'debug': True,
    'entire_tile': False,
    'nodes': 1,
    'node_memory': '50G',
    'chunk_size': {'x': 512, 'y': 512},
}
```

### Tiles mode with spatial erosion
```python
ProdParams = {
    'sensor': 'HLS_SR',
    'unit': 2,
    'nbYears': -1,
    'regions': './sites.kml',
    'file_variables': ['SiteID', 'AsssD_1', 'AsssD_2'],
    'mode': 'tiles',
    's2_grid_path': 'sentinel-2-grid.parquet',
    'subdivide_tiles': False,
    'spatial_buffer_m': -20,                            # erode polygons 20m before tile lookup
    'temporal_buffer': [["2024-06-01", "2024-08-31"]],
    'resolution': 30,
    'projection': 'EPSG:3979',
    'prod_names': ['mosaic', 'LAI'],
    'out_folder': './output',
    'out_datatype': 'int16',
    'output_type': 'geotiff',
}
```

### Multi-year monthly mosaics (no dates in file)
```python
ProdParams = {
    'sensor': 'S2_SR',
    'unit': 2,
    'nbYears': -1,
    'year': 2020,
    'months': [6, 7, 8],
    'num_years': 3,                     # generates Jun–Aug for 2020, 2021, 2022
    'regions': './sites.kml',
    'file_variables': ['SiteID', None, None],  # no date columns in file
    'mode': 'regions',
    'resolution': 10,
    'projection': 'EPSG:3979',
    'prod_names': ['mosaic'],
    'out_folder': './output',
    'out_datatype': 'int16',
    'output_type': 'geotiff',
}
```

### CSV export
```python
ProdParams = {
    ...
    'output_type': 'csv',
    'csv_scale': 30,
    'csv_dropNulls': True,
    'csv_max_pixels': 50_000,   # lower for faster test runs
}
```

---

## Processing Log

After polygon validation, a CSV is written to `out_folder/polygon_processing_log.csv`:

```
region_id, date,       area_m2,      will_process, status,  skip_reason
region0,   2023-06-30, 0.00,         False,        SKIPPED, All coordinates identical
region20,  2023-06-30, 12500000.00,  True,         QUEUED,  Valid polygon
```

---

## Parameter Validation

`prepare_production_params()` runs validation as Step 1/4 before any file I/O. It returns `None` and prints a numbered error list on failure.

| Field | Rule |
|-------|------|
| `regions` | `.kml`, `.shp`, or dict; file must exist on disk |
| `file_variables` | Exactly 3 elements; first (ID) cannot be `None`; date elements can be `None` |
| `regions_start_index` | Integer ≥ 0 |
| `regions_end_index` | Integer ≥ `regions_start_index`, or `None` |
| `spatial_buffer_m` | Number (int or float) |
| `temporal_buffer` | List of `[int, int]` pairs OR `["YYYY-MM-DD", "YYYY-MM-DD"]` pairs — cannot mix types; end date must be after start date in override mode |
| `num_years` | Integer ≥ 1 |
| `mode` | `'regions'` or `'tiles'` |

---

## Troubleshooting

**Wrong region IDs (region0, region1 instead of region20, region39)**
→ `file_variables[0]` is the wrong column. `TARGET_FID` is a row counter — use your actual site ID column (e.g., `SiteID`)

**Region-specific dates not applied**
→ Check that your KML has date attribute columns; verify column names match `file_variables[1]` and `[2]`; dates must be `YYYY-MM-DD`

**All polygons skipped / zero area**
→ Check `polygon_processing_log.csv`; if using `spatial_buffer_m`, the negative value may be too large and collapsing polygons; verify file contains polygons (not just points)

**Empty regions after tiles mode**
→ Verify `s2_grid_path` parquet exists and spatially covers your regions

**Multiple temporal buffer pairs ignored for global windows**
→ Multiple offset pairs only expand region-specific dates (from file). For global `year`/`months` windows, use a single pair (e.g., `[[-5, 5]]`)

**Dates swapped automatically**
→ Intentional — if `start_date > end_date`, they are swapped with a console warning

**`result` is `None`**
→ Either validation failed (read the printed error list) or all polygons are zero-area (check `polygon_processing_log.csv`)

---

> **Key rule:** Always set `file_variables` to match your KML/SHP column names exactly. Use `['MyID', None, None]` if the file has no date attributes.