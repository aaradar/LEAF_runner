# Afforestation 10m
ProdParams = {
    # ============ REQUIRED PARAMETERS ============
    'sensor': 'S2_SR',                    # Sensor type
    'unit': 2,                   # A data unit code (1 or 2 for TOA or surface reflectance)    
    'nbYears': -1,            # positive int for annual product, or negative int for monthly product
    #'year': 2023,                          # CHANGE THIS: Year to process
    #'months': [8,9,10],                   # CHANGE THIS: Months to process (1-12)
    
    
    # ============ REGION PARAMETERS ============
    'regions': kml,  # CHANGE THIS: Path to your KML/SHP file
    'file_variables': ['TARGET_FID', 'AsssD_1', 'AsssD_2'], # CHANGE THIS: id, start_date, end_date (None for dates if they don't exist)
    #'file_variables': ['id', 'begin', 'end'],
    #'file_variables': ['system:index', None, None],
    'regions_start_index': 3000,              # CHANGE THIS: Start at this region index
    'regions_end_index': 3999,             # CHANGE THIS: End at this index (None = all)
    
    # ============ BUFFER PARAMETERS ============
    #'spatial_buffer_m': 0,             # UNCOMMENT & CHANGE: Buffer in meters around regions
    #'temporal_buffer': [[50, 90], [-10, 10]],          # UNCOMMENT & CHANGE: [days_before, days_after]
    'temporal_buffer': [["2025-08-01", "2025-08-31"]],
    #'num_years': 3,
    
    
    # ============ OUTPUT PARAMETERS ============
    'resolution': 10,                      # Resolution in meters
    'projection': 'EPSG:3979',             # Coordinate projection
    'prod_names': ['mosaic'],    #['LAI', 'FCOVER', 'fAPAR', 'Albedo'], 
    #'bands': ['blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir08', 'swir16', 'swir22'],
    'out_folder': 'E:/S2_mosaics_runner_2026/benchmark/S2_3000-3999_august_2025_10m',
    'out_datatype': 'int16'
}

# Start 2026-02-25 at 12:47pm
# End 2026-02-26 at 7:10am

CompParams = {
  "number_workers":32,     
  "debug"       : True,
  "entire_tile" : False, 
  "nodes"       : 1,
  "node_memory" : "16G",
  'chunk_size': {'x': 512, 'y': 512}
}

# Afforestation 20m
ProdParams = {
    # ============ REQUIRED PARAMETERS ============
    'sensor': 'S2_SR',                    # Sensor type
    'unit': 2,                   # A data unit code (1 or 2 for TOA or surface reflectance)    
    'nbYears': -1,            # positive int for annual product, or negative int for monthly product
    #'year': 2023,                          # CHANGE THIS: Year to process
    #'months': [8,9,10],                   # CHANGE THIS: Months to process (1-12)
    
    
    # ============ REGION PARAMETERS ============
    'regions': kml,  # CHANGE THIS: Path to your KML/SHP file
    'file_variables': ['TARGET_FID', 'AsssD_1', 'AsssD_2'], # CHANGE THIS: id, start_date, end_date (None for dates if they don't exist)
    #'file_variables': ['id', 'begin', 'end'],
    #'file_variables': ['system:index', None, None],
    'regions_start_index': 3000,              # CHANGE THIS: Start at this region index
    'regions_end_index': 3999,             # CHANGE THIS: End at this index (None = all)
    
    # ============ BUFFER PARAMETERS ============
    #'spatial_buffer_m': 0,             # UNCOMMENT & CHANGE: Buffer in meters around regions
    #'temporal_buffer': [[50, 90], [-10, 10]],          # UNCOMMENT & CHANGE: [days_before, days_after]
    'temporal_buffer': [["2025-08-01", "2025-08-31"]],
    #'num_years': 3,
    
    
    # ============ OUTPUT PARAMETERS ============
    'resolution': 10,                      # Resolution in meters
    'projection': 'EPSG:3979',             # Coordinate projection
    'prod_names': ['mosaic'],    #['LAI', 'FCOVER', 'fAPAR', 'Albedo'], 
    #'bands': ['blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir08', 'swir16', 'swir22'],
    'out_folder': 'E:/S2_mosaics_runner_2026/benchmark/S2_3000-3999_august_2025_10m',
    'out_datatype': 'int16'
}

# Start 2026-02-25 at 8:30am
# End 2026-02-26 at 

CompParams = {
  "number_workers":32,     
  "debug"       : True,
  "entire_tile" : False, 
  "nodes"       : 1,
  "node_memory" : "16G",
  'chunk_size': {'x': 512, 'y': 512}
}


# Pipeline region1 10m
# Pipeline region1 20m