# Parameter Preparation System - README

## Installation

**Step 0: Create and activate a Conda environment**
```bash
conda create -n leaf-env python=3.11.14
conda activate leaf-env
```

**Step 1: Core dependencies**
```bash
conda install -c conda-forge click==8.1.7 dask==2024.5.2 dask-jobqueue==0.9.0 \
  numpy==1.24.4 odc-geo==0.4.8 odc-stac==0.3.10 pandas==2.2.3 psutil==5.9.8 \
  pyproj==3.6.1 pystac-client==0.8.2 rasterio==1.3.10 Requests==2.32.3 \
  rioxarray==0.15.6 stackstac==0.5.1 tqdm==4.66.4 urllib3==2.3.0 \
  xarray==2024.6.0 "bokeh!=3.0.*,>=2.4.2" gdal
```

**Step 2: GeoPandas and spatial libraries**
```bash
conda install -c conda-forge geopandas shapely pyogrio packaging
```

**Step 3: Connect the Conda environment to Jupyter**
```bash
conda install ipykernel
python -m ipykernel install --user --name leaf-env --display-name "Python (leaf-env)"
```

**Step 4: Run notebooks**
```bash
jupyter notebook
# or
jupyter lab
```

## Overview

This README documents the new Parameter Preparation System added to the LEAF production pipeline. The system provides pre-production parameter validation, temporal window generation, region loading from files, and polygon validation capabilities.

## New Files Added

**New Files:**
- `prepare_params.py` - Starter functions for Mosiac Production Modularity
- `parameter_preparation.ipynb` — KML-driven production example
- `test_regions_kml.py` - Output validation tests for kml files
- `LEAF_runner.png` - Flowchart of the new/updated files
- `source/leaf_wrapper.py` — Region conversion module
- `source/polygon_validator.py` — Zero-area polygon detection and validation
- `Sample Points/AfforestationSItesFixed.kml` — Example KML file with afforestation polygons
- `Sample Points/FieldPoints32_2018.shp` — Example Shapefile with point geometries
- `Sample Points/GTA.kml` — kml created from with GE
- `Sample Points/ShapeTest.py` — kml/shp to csv testing
- `testing/test_leaf_wrapper.py` — Output validation tests for both KML and SHP
- `testing/test_negative_buffer.py` — Output validation tests for buffers and polygon integrity

**Updated Files:**
- `Production.py` — kml date handling
- `requirements.txt` — Two-step GeoPandas installation

---

### 1. **prepare_params.py** - Main Parameter Preparation Module
**Location:** `./prepare_params.py`

**Purpose:** Orchestrates all parameter processing and validation before calling `Production.py main()`.

**Key Features:**
- 4 temporal window generation modes
- KML/Shapefile region loading
- Temporal and spatial buffering
- Date symmetry handling
- Polygon validation and filtering

### 2. **polygon_validator.py** - Polygon Validation Module
**Location:** `./source/polygon_validator.py`

**Purpose:** Filters zero-area polygons and creates processing logs.

**Key Features:**
- Zero-area polygon detection
- Area calculation (with/without Shapely)
- Processing log generation
- CSV export of validation results

### 3. **leaf_wrapper.py** - Region File Handler
**Location:** `./source/leaf_wrapper.py`

**Purpose:** Reads KML/Shapefile files and converts them to LEAF-compatible region dictionaries.

**Key Features:**
- KML and Shapefile support
- Spatial buffering (meters)
- Temporal buffer extraction from file attributes
- Point-to-polygon conversion
- Z-coordinate removal (3D → 2D)

### 4. **Production.py** - Updated Main Production Script
**Location:** `./Production.py`

**Purpose:** Main production orchestrator with region-specific date support.

**Key Changes:**
- Added region-specific start/end date handling
- Temporal buffer integration
- Enhanced region processing loop

### 5. **parameter_preparation.ipynb** - Interactive Notebook
**Location:** `./parameter_preparation.ipynb`

**Purpose:** Jupyter notebook with 4 pre-configured parameter templates for different use cases.

**Templates:**
1. Single Year Monthly Mosaics
2. Custom Date Ranges
3. Single Date Windows
4. Multi-Year Monthly Mosaics

---

## System Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT                                │
│  • ProdParams (sensor, year, months, regions)               │
│  • CompParams (workers, memory, debug)                       │
│  • KML/SHP files (optional)                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              prepare_params.py                               │
│         prepare_production_params()                          │
└─────────────────────────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │ Step 1 │  │ Step 2 │  │ Step 3 │
    │ Regions│  │ Dates  │  │Symmetry│
    └────────┘  └────────┘  └────────┘
         │           │           │
         │           │           │
         └───────────┼───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │       Step 4          │
         │  Polygon Validation   │
         └───────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Validated Params    │
         │   + Processing Log    │
         └───────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Production.py       │
         │      main()           │
         └───────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Mosaic Generation   │
         └───────────────────────┘
```

---

## Function Connection Map

### Connection Map

```
prepare_params.py
├── imports leaf_wrapper.py
│   └── regions_from_kml()
│       └── LeafWrapper class
│           ├── load() - reads KML/SHP
│           ├── _apply_buffer() - spatial buffering
│           └── to_region_dict() - converts to LEAF format
│
├── imports polygon_validator.py
│   ├── filter_valid_polygons() - removes zero-area polygons
│   ├── create_processing_log() - generates CSV logs
│   ├── is_zero_area_polygon() - validation logic
│   └── calculate_polygon_area() - area computation
│
└── provides functions to Production.py
    ├── prepare_production_params() - main orchestrator
    ├── handle_regions_from_file() - region loading
    ├── form_time_windows() - 4 temporal modes
    ├── ensure_date_symmetry() - date validation
    └── validate_and_filter_polygons() - polygon filtering

Production.py
├── imports prepare_params.py
├── calls prepare_production_params() (optional)
└── main() - receives validated parameters
    ├── Uses region_start_dates (if available)
    ├── Uses region_end_dates (if available)
    └── Falls back to default start_dates/end_dates

parameter_preparation.ipynb
├── imports prepare_params.py
└── provides 4 templates for different use cases
```

---

## Function Reference

### prepare_params.py Functions

#### `prepare_production_params(ProdParams, CompParams)`
**Main orchestration function**

**Steps:**
1. Load regions from KML/SHP files (if applicable)
2. Generate temporal windows (4 modes)
3. Ensure date symmetry
4. Validate and filter polygons

**Returns:**
```python
{
    'ProdParams': validated_params,
    'CompParams': comp_params,
    'processing_log': dataframe_or_list
}
```

**Returns `None` if validation fails (e.g., all polygons are zero-area)**

---

#### `handle_regions_from_file(ProdParams)`
**Handles KML/Shapefile-based region input**

**What it does:**
1. Detects if `regions` is a file path (.kml or .shp)
2. Calls `leaf_wrapper.regions_from_kml()` to load regions
3. Applies temporal buffer to region-specific dates
4. Updates ProdParams with loaded regions

**Parameters used:**
- `ProdParams['regions']` - path to KML/SHP file
- `ProdParams['regions_start_index']` - start position
- `ProdParams['regions_end_index']` - end position
- `ProdParams['spatial_buffer_m']` - spatial buffer in meters
- `ProdParams['temporal_buffer']` - [days_before, days_after]

**Returns:** Updated ProdParams with regions dictionary

---

#### `form_time_windows(ProdParams)`
**Creates start/end dates based on 4 temporal modes**

**Mode 1: Single Year Monthly Mosaics**
```python
ProdParams = {
    'year': 2023,
    'months': [6, 7, 8]
}
# Generates:
# start_dates = ['2023-06-01', '2023-07-01', '2023-08-01']
# end_dates = ['2023-06-30', '2023-07-31', '2023-08-31']
```

**Mode 2: Custom Date Ranges**
```python
ProdParams = {
    'start_dates': ['2023-06-01', '2023-08-15'],
    'end_dates': ['2023-06-30', '2023-09-15']
}
# Uses dates as-is
```

**Mode 3: Single Date Auto-Generation**
```python
ProdParams = {
    'start_date': '2023-06-01'
    # end_date is auto-generated to match
}
# OR
ProdParams = {
    'end_date': '2023-06-30'
    # start_date is auto-generated to match
}
```

**Mode 4: Multi-Year Monthly Mosaics**
```python
ProdParams = {
    'year': 2020,
    'months': [6, 7],
    'num_years': 3
}
# Generates:
# 2020-06, 2020-07, 2021-06, 2021-07, 2022-06, 2022-07
```

**Temporal Buffer Application:**
If `temporal_buffer` is specified, it's applied to all dates:
```python
ProdParams['temporal_buffer'] = [-5, 5]  # [days_before, days_after]
# '2023-06-01' becomes '2023-05-27' (start)
# '2023-06-30' becomes '2023-07-05' (end)
```

---

#### `ensure_date_symmetry(ProdParams)`
**Ensures start_dates and end_dates are symmetric**

**What it does:**
- If only `start_dates` exists → copies to `end_dates`
- If only `end_dates` exists → copies to `start_dates`
- If both exist → no change

---

#### `validate_and_filter_polygons(ProdParams)`
**Validates and filters zero-area polygons**

**What it does:**
1. Creates processing log CSV
2. Filters out invalid polygons
3. Updates ProdParams with only valid regions
4. Filters region-specific dates to match valid regions

**Returns:**
- `(ProdParams, processing_log)` if valid polygons exist
- `(None, processing_log)` if all polygons are invalid

**Processing Log CSV Columns:**
- `region_id` - Region identifier
- `date` - Processing date
- `area_m2` - Polygon area in square meters
- `will_process` - Boolean (True/False)
- `status` - 'QUEUED' or 'SKIPPED'
- `skip_reason` - Why polygon was skipped
- `timestamp` - When log entry was created

---

### polygon_validator.py Functions

#### `calculate_polygon_area(coordinates)`
**Calculates polygon area in square meters**

**Uses:**
- Shapely (if available) - preferred method
- Shoelace formula (fallback) - works without Shapely

**Returns:** Area in square meters (approximate for WGS84)

---

#### `is_zero_area_polygon(region_data)`
**Checks if polygon has zero or near-zero area**

**Checks for:**
1. All coordinates identical (degenerate polygon)
2. Fewer than 3 unique points
3. Area below threshold (default: 1 m²)

**Returns:** `(is_zero_area, area_m2, reason)`

---

#### `filter_valid_polygons(regions)`
**Filters out zero-area polygons**

**Returns:** `(valid_regions_dict, validation_log)`

---

#### `create_processing_log(regions, end_dates, output_path)`
**Creates comprehensive processing log CSV**

**Columns:**
- region_id, date, area_m2, will_process, status, skip_reason, timestamp

**Saves to:** CSV file at `output_path`

---

### leaf_wrapper.py Functions

#### `regions_from_kml(kml_file, start, end, prefix, spatial_buffer_m)`
**Load KML/Shapefile and return polygon regions**

**Parameters:**
- `kml_file` - Path to KML or Shapefile
- `start` - Starting position (0-based, inclusive)
- `end` - Ending position (0-based, inclusive, None = all)
- `prefix` - Prefix for region names (default: "region")
- `spatial_buffer_m` - Buffer size in meters (can be negative)

**Returns:** `(regions_dict, region_start_dates, region_end_dates)`

**Example:**
```python
regions, starts, ends = regions_from_kml(
    'sites.kml',
    start=0,
    end=5,
    spatial_buffer_m=-20  # Negative buffer = erosion
)
# Returns regions 0-5 with 20m erosion applied
```

---

#### `LeafWrapper` Class

**Constructor:**
```python
wrapper = LeafWrapper(polygon_file, spatial_buffer_m=None)
```

**Methods:**

##### `load()`
Loads KML or Shapefile into GeoDataFrame
- Drops Z coordinates (3D → 2D)
- Applies spatial buffer if specified
- Returns `self` for chaining

##### `_apply_buffer()`
Applies spatial buffer to geometries
- Converts to EPSG:3979 (Canada Atlas Lambert)
- Applies metric buffer
- Handles negative buffers (creates zero-area points at centroid)
- Converts back to original CRS

##### `to_region_dict(use_target_fid=True)`
Converts geometries to LEAF-compatible dictionary

**Extracts:**
- Region coordinates
- `TARGET_FID`, `SiteID`, or `pointid` as region ID
- `AsssD_1` as start_date
- `AsssD_2` through `AsssD_7` as end_date (first non-null)

**Returns:**
```python
{
    'region_id': {
        'r_id': region_id,
        'coordinates': [[[x1, y1], [x2, y2], ...]],
        'start_date': 'YYYY-MM-DD',  # From AsssD_1
        'end_date': 'YYYY-MM-DD'     # From AsssD_2-7
    }
}
```

---

### Production.py Changes

#### Enhanced Region Processing Loop

**Before:**
```python
for reg_name in region_names:
    usedParams = eoPM.set_spatial_region(usedParams, reg_name)
    for TIndex in range(nTimes):
        usedParams = eoPM.set_current_time(usedParams, TIndex)
        eoMz.one_mosaic(usedParams, CompParams)
```

**After (with region-specific dates):**
```python
for reg_name in region_names:
    usedParams = eoPM.set_spatial_region(usedParams, reg_name)
    
    # Check for region-specific dates
    if has_region_dates and reg_name in usedParams['region_start_dates']:
        # Use region-specific dates
        region_start_dates = usedParams['region_start_dates'][reg_name]
        region_end_dates = usedParams['region_end_dates'].get(reg_name)
        usedParams['start_dates'] = region_start_dates
        usedParams['end_dates'] = region_end_dates
        nTimes = 1
    else:
        # Use default dates
        usedParams['start_dates'] = default_start_dates
        usedParams['end_dates'] = default_end_dates
        nTimes = len(default_start_dates)
    
    for TIndex in range(nTimes):
        usedParams = eoPM.set_current_time(usedParams, TIndex)
        eoMz.one_mosaic(usedParams, CompParams)
```

---

## Usage Examples

### Example 1: Basic Usage

```python
from prepare_params import prepare_production_params
from Production import main

ProdParams = {
    'sensor': 'HLS_SR',
    'year': 2023,
    'months': [6, 7, 8],
    'regions': './data/sites.kml',
    'regions_start_index': 0,
    'regions_end_index': 10,
    'resolution': 30,
    'projection': 'EPSG:3979',
    'out_folder': './output'
}

CompParams = {'number_workers': 10, 'debug': True}

result = prepare_production_params(ProdParams, CompParams)
if result:
    main(result['ProdParams'], result['CompParams'])
```

### Example 2: Spatial and Temporal Buffers

```python
ProdParams = {
    'sensor': 'S2_SR',
    'year': 2023,
    'months': [6, 7],
    'regions': './sites.kml',
    'spatial_buffer_m': -20,        # 20m erosion
    'temporal_buffer': [-7, 7],     # 7 days before/after
    'resolution': 30,
    'out_folder': './output'
}
```

### Example 3: Multi-Year Processing

```python
ProdParams = {
    'sensor': 'HLS_SR',
    'year': 2020,
    'months': [6, 7, 8],
    'num_years': 3,                 # Process 2020-2022
    'regions': './sites.kml',
    'resolution': 30,
    'out_folder': './output'
}
# Generates 9 mosaics: 2020-06/07/08, 2021-06/07/08, 2022-06/07/08
```

### Example 4: Custom Date Ranges

```python
ProdParams = {
    'sensor': 'HLS_SR',
    'start_dates': ['2023-06-01', '2023-08-15'],
    'end_dates': ['2023-06-30', '2023-09-15'],
    'regions': './sites.kml',
    'resolution': 30,
    'out_folder': './output'
}
```

---

## Processing Log Example

After running `validate_and_filter_polygons()`, a CSV log is created:

```csv
region_id,date,area_m2,will_process,status,skip_reason,timestamp
region0,2023-06-30,0.00,False,SKIPPED,All coordinates identical (point not polygon),2025-02-09T10:30:00
region1,2023-06-30,0.00,False,SKIPPED,Only 1 unique coordinate(s),2025-02-09T10:30:00
region2,2023-06-30,12500000.00,True,QUEUED,Valid polygon,2025-02-09T10:30:00
region3,2023-06-30,8750000.00,True,QUEUED,Valid polygon,2025-02-09T10:30:00
```

**Console Output:**
```
Summary:
   Total region-date combinations: 4
   Valid for processing: 2
   Skipped (zero area): 2

Zero-area regions detected:
   - region0
   - region1

Proceeding with 2 valid region(s) out of 4 total
```

---

## Error Handling

### All Polygons Invalid

```python
result = prepare_production_params(ProdParams, CompParams)

if result is None:
    print("ERROR: All polygons have zero area!")
    # Check processing log for details
    # Typical causes:
    # 1. All coordinates are identical (points, not polygons)
    # 2. Negative buffer too large (collapses polygons)
    # 3. Invalid KML/SHP file format
```

### File Not Found

```python
# If KML file doesn't exist:
"""
ERROR: Region file not found: ./sites.kml
Current working directory: /home/user/project
Please provide a valid file path.
"""
```

### Invalid Date Range

```python
# If start_date > end_date, dates are automatically swapped:
"""
<form_time_windows> Swapping dates for window 0: 
  2023-08-01 <-> 2023-06-01
"""
```

## Troubleshooting

**All polygons skipped**
- Check processing log for skip reasons
- Reduce negative buffer size
- Verify KML/SHP file contains valid polygons

**Region-specific dates not working**
- Ensure KML has `AsssD_1` (start date) attribute
- Ensure KML has `AsssD_2` through `AsssD_7` (end date) attributes
- Check Sentinel-2 or Landsat has data for the specified dates
- Dates must be in 'YYYY-MM-DD' format

**Dates swapped automatically**
- This is intentional when start_date > end_date
- Check console output for swap messages

---

## Summary

The Parameter Preparation System provides automated parameter processing, validation, and region loading for LEAF production. Key capabilities include flexible temporal modes, spatial/temporal buffering, polygon validation, and comprehensive error logging.

---