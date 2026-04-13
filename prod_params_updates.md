
# Updates for Leaf Runner
ProdParams = {
        # ============ REQUIRED PARAMETERS ============
        # You don't need to use months or start and end dates old date modes if using kml file 

        'sensor': 'HLS_SR',          # A sensor type string (e.g., 'S2_SR', 'HLS_SR', 'HLSL30_SR', 'HLSS30_SR' or 'MOD_SR')
        'unit': 2,                   # A data unit code (1 or 2 for TOA or surface reflectance)
        'nbYears': -1,               # positive int for annual product, or negative int for monthly product
        #'year': 2023,               # CHANGE THIS: Year to process according to a month
        #'months': [8, 9, 10],       # CHANGE THIS: Months to process (1-12)

        # ============ REGION PARAMETERS ============
        # Always provide a region, file_variables, region start and end dates
        # Add tile variables if using

        'regions': kml,              # CHANGE THIS: Path to your KML/SHP file for the regions you want to run
        's2_grid_path': s2_mgrs,     # CHANGE THIS: Path to your mgrs kml/parquet file for tiling grid to run for tile mode
        'mode': 'regions',           # default: regions, either 'regions' or 'tiles' mode
        'subdivide_tiles': False,    # Whether to subdivide tiles into smaller regions (only applicable if mode is 'tiles')
        'file_variables': ['TARGET_FID', 'AsssD_1', 'AsssD_2'],  # CHANGE THIS: id, start_date, end_date (None for dates if they don't exist)
        #'file_variables': ['id', 'begin', 'end'], # different options
        #'file_variables': ['system:index', None, None], # different kml file keys
        'regions_start_index': 3000, # CHANGE THIS: Start querying at this region index
        'regions_end_index': 3002,   # CHANGE THIS: End at this index (inclusive) (None = all)

        # ============ BUFFER PARAMETERS ============
        # You don't need a spatial buffer, usually only temporal buffer

        #'spatial_buffer_m': -10,                          # Buffer in meters around all regions 

        #'temporal_buffer': [[50, 90]],         # mode 1: [days_before, days_after]
        #'temporal_buffer': [[50, 90], [-10, 10]],         # mode 1: multi [days_before, days_after]
        'temporal_buffer': [["2025-08-01", "2025-08-31"]], # mode 2: set start and end dates for all regions (can do more than one)
        #'temporal_buffer': [["2025-08-01", "2025-08-31"], [["2025-09-01", "2025-09-31"], [["2025-10-01", "2025-10-31"]], # mode 2: multi set start and end dates for all regions
        #'num_years': 10, # repeat each the start and end dates in each regions list n times incrementing the year

        # ============ OUTPUT PARAMETERS ============
        'resolution': 30,            # Resolution in meters
        'projection': 'EPSG:3979',   # Coordinate projection
        'IncludeAngles': False,
        'prod_names': ['LAI'],       # ['mosaic', 'LAI', 'fCOVER', 'fAPAR', 'Albedo']
        'out_folder': r'E:\Testing1\hls_lai_10m', # output location
        'out_datatype': 'int16',

        # ============ CSV OUTPUT ============
        # Default GeoTIFF 

        'output_type': 'geotiff',    # switch to 'csv' to export as CSV instead of GeoTIFF
        'csv_scale': 10,             # optional: same as resolution, explicit here for clarity
        'csv_dropNulls': True,       # optional: skip masked/nodata pixels (mirrors GEE dropNulls)
        'csv_max_pixels': 100_000,   # optional: lower this first for a quick test
    }
