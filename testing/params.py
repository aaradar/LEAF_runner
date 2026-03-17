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
# total time 18 hours and 23 minutes

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
# End 2026-02-25 at 11:14pm
# total time 14:44 hours

CompParams = {
  "number_workers":32,     
  "debug"       : True,
  "entire_tile" : False, 
  "nodes"       : 1,
  "node_memory" : "16G",
  'chunk_size': {'x': 512, 'y': 512}
}

# 1000 regions in afforestation comparison
# 18:23 hours for 10m and 14:44 hours for 20m. The 20m is about 80% of the time of the 10m
# dask.compute()



# Pipeline region1 10m
# TEMPLATE 1: Single Year Monthly Mosaics
# Use Case: Generate monthly mosaics for specific months in one year
# Example: Summer months (June, July, August) of 2023
# Customize these parameters as needed:

ProdParams = {
    # ============ REQUIRED PARAMETERS ============
    'sensor': 'S2_SR',                    # Sensor type
    'unit': 2,                   # A data unit code (1 or 2 for TOA or surface reflectance)    
    'nbYears': -1,            # positive int for annual product, or negative int for monthly product
    #'year': 2023,                          # CHANGE THIS: Year to process
    #'months': [8,9,10],                   # CHANGE THIS: Months to process (1-12)
    
    
    # ============ REGION PARAMETERS ============
    'regions': tmx,  # CHANGE THIS: Path to your KML/SHP file
    'mode': 'regions',
    #'file_variables': ['TARGET_FID', 'AsssD_1', 'AsssD_2'], # CHANGE THIS: id, start_date, end_date (None for dates if they don't exist)
    'file_variables': ['id', 'begin', 'end'],
    #'file_variables': ['system:index', None, None],
    'regions_start_index': 0,              # CHANGE THIS: Start at this region index
    'regions_end_index': 0,             # CHANGE THIS: End at this index (None = all)
    
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
    'out_folder': 'E:/S2_mosaics_runner_2026/benchmark/S2_pipeline2_august_32w_16gb_10m',
    'out_datatype': 'int16'
}

CompParams = {
  "number_workers":32,     
  "debug"       : True,
  "entire_tile" : False, 
  "nodes"       : 1,
  "node_memory" : "16G",
  'chunk_size': {'x': 512, 'y': 512}
}

#runs into problems with memory
# crashes due to 16GB memory limit per worker being overloaded at completion of the dask task
# times would be around:
# 10:27 to 1:02 = 2 hours and 25 minutes for 1 region at 10m resolution with 32 workers and 16GB memory per worker
# 11:19 to 1:40 = 2 hours and 21 minutes for 1 region at 10m resolution with 10 workers and 50GB memory per worker
CompParams = {
  "number_workers":10,     
  "debug"       : True,
  "entire_tile" : False, 
  "nodes"       : 1,
  "node_memory" : "50G",
  'chunk_size': {'x': 512, 'y': 512}
}
# 8:54 - 11:17 = 2 hours and 23 minutes for 1 region at 10m resolution with 10 workers and 50GB memory per worker
# 12:22 to 14:41 = 2 hours and 19 minutes for 1 region at 10m resolution with 10 workers and 50GB memory per worker

# Pipeline region1 20m

# TEMPLATE 1: Single Year Monthly Mosaics
# Use Case: Generate monthly mosaics for specific months in one year
# Example: Summer months (June, July, August) of 2023
# Customize these parameters as needed:

ProdParams = {
    # ============ REQUIRED PARAMETERS ============
    'sensor': 'S2_SR',                    # Sensor type
    'unit': 2,                   # A data unit code (1 or 2 for TOA or surface reflectance)    
    'nbYears': -1,            # positive int for annual product, or negative int for monthly product
    #'year': 2023,                          # CHANGE THIS: Year to process
    #'months': [8,9,10],                   # CHANGE THIS: Months to process (1-12)
    
    
    # ============ REGION PARAMETERS ============
    'regions': tmx,  # CHANGE THIS: Path to your KML/SHP file
    'mode': 'regions',
    #'file_variables': ['TARGET_FID', 'AsssD_1', 'AsssD_2'], # CHANGE THIS: id, start_date, end_date (None for dates if they don't exist)
    'file_variables': ['id', 'begin', 'end'],
    #'file_variables': ['system:index', None, None],
    'regions_start_index': 0,              # CHANGE THIS: Start at this region index
    'regions_end_index': 0,             # CHANGE THIS: End at this index (None = all)
    
    # ============ BUFFER PARAMETERS ============
    #'spatial_buffer_m': 0,             # UNCOMMENT & CHANGE: Buffer in meters around regions
    #'temporal_buffer': [[50, 90], [-10, 10]],          # UNCOMMENT & CHANGE: [days_before, days_after]
    'temporal_buffer': [["2025-08-01", "2025-08-31"]],
    #'num_years': 3,
    
    
    # ============ OUTPUT PARAMETERS ============
    'resolution': 20,                      # Resolution in meters
    'projection': 'EPSG:3979',             # Coordinate projection
    'prod_names': ['mosaic'],    #['LAI', 'FCOVER', 'fAPAR', 'Albedo'], 
    #'bands': ['blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir08', 'swir16', 'swir22'],
    'out_folder': 'E:/S2_mosaics_runner_2026/benchmark/S2_pipeline_august_32w_16gb_20m',
    'out_datatype': 'int16'
}

CompParams = {
  "number_workers":32,     
  "debug"       : True,
  "entire_tile" : False, 
  "nodes"       : 1,
  "node_memory" : "16G",
  'chunk_size': {'x': 512, 'y': 512}
}

#convex hull of all polygons
# for all tile polygons: clip(tiles polygon, convex hull)
# clip to convex hull of all polygons

# bonney mitchell
# what tile grid
 
#10:24 to 11:16 = 52 minutes for 1 region at 20m resolution with 32 workers and 16GB memory per worker


CompParams = {
  "number_workers":10,     
  "debug"       : True,
  "entire_tile" : False, 
  "nodes"       : 1,
  "node_memory" : "50G",
  'chunk_size': {'x': 512, 'y': 512}
}

# 11:25 to 12:14 = 49 minutes for 1 region at 20m resolution with 10 workers and 50GB memory per worker.


#tiles
#one tile
#147 minutes = 2 hours and 27 minutes for one tile at 10m res with 10 workers and 50GB memory per worker


#clipping tiles

# tiles 50 regions (3000-3049) at 10m res with 10 workers and 50GB memory per worker
# 8:58 to 9:34 = 36 mins


# regions 50 regions (3000-3049) at 10m res with 10 workers and 50GB memory per worker
# 9:44 to 10:26 = 42 mins

#clipping tiles
# 1000 regions (3000-3999) at 10m res with 10 workers and 50GB memory per worker
# 2:52 pm to 1:36 am = 10 hours and 44 minutes


#hls 2026-03-12

# 50 regions at 30m HLS_SR afforestation
#  12:26pm to 1:48pm = 1 hour and 22 minutes 
# for 50 regions at 30m res with 10 workers and 50GB memory per worker

#1000 regions at 30m HLS_SR afforestation
# 1:52pm to 6:40 am + 9:19am to 9:04pm = 16 hours and 52 minutes 
# for 1000 regions at 30m res with 10 workers and 50GB memory per worker

# 50 regions with tiles
# 7:30 am to 8:32 am = 1 hour and 2 minutes

# 1000 regions with tiles
# 8:35 am