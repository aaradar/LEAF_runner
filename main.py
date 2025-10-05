import os
#import re
import sys
#import subprocess
#import argparse
from pathlib import Path


#Get the absolute path to the parent of current working directory 
cwd    = os.getcwd()
source_path = os.path.join(cwd, 'source')
sys.path.append(source_path)
if str(Path(__file__).parents[0]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[0]))
if str(Path(__file__).parents[0]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[0]))

import source.eoMosaic as eoMz
import source.eoParams as eoPM
#import source.eoTileGrids as eoTG
import source.LEAFProduction as leaf


def gdal_mosaic_rasters(sorted_files_to_mosaic:list, merge_output_file:str):
        
    options = ['-of', 'GTiff', '-v']
    gdal_merge_command = ['gdal_merge.py', '-o', merge_output_file] + options + sorted_files_to_mosaic
    os.system(" ".join(gdal_merge_command))





#############################################################################################################
# Description: This is the main function for generating composite images 
#############################################################################################################
def main():
  #==========================================================================================================
  # The following two lines are for two ways to obtaining input parameters
  #==========================================================================================================
  prod_params, comp_params = eoPM.form_inputs(ProdParams, CompParams)  # Using two dictionaries to input required parameters
  #prod_params, comp_params = eoPM.form_inputs()                       # Using command options to input required parameters   
  if prod_params is None or comp_params is None:
    print('<main> Incomplete input parameters (two dictionaries, ProdParams and CompParams, are required)!')
    return
  
  #==========================================================================================================
  # Determine which product is required
  #==========================================================================================================
  prod_type = eoPM.which_product(ProdParams)
  if 'veg' in prod_type:
    leaf.LEAF_production(prod_params, comp_params)

  else:
    eoMz.MosaicProduction(prod_params, comp_params)
    


ottawa_region = {
    'type': 'Polygon',
    'coordinates': [
       [
         [-76.120,45.184], 
         [-75.383,45.171],
         [-75.390,45.564], 
         [-76.105,45.568], 
         [-76.120,45.184]
       ]
    ]
}

CompParams = {
  "debug"       : True,
  "entire_tile" : False,     #
  "nodes"       : 1,
  "node_memory" : "14G",
  "number_workers" : 10
}

  
ProdParams = {
    'sensor': 'S2_SR',       # A sensor type string (e.g., 'S2_SR' or 'HLSS30_SR' or 'MOD_SR')
    'unit': 2,                   # A data unit code (1 or 2 for TOA or surface reflectance)    
    'year': 2024,                # An integer representing image acquisition year
    'nbYears': -1,               # positive int for annual product, or negative int for monthly product
    'months': [8],            # A list of integers represening one or multiple monthes     
    'tile_names': ['tile55_933'], # A list of (sub-)tile names (defined using CCRS' tile griding system) 
    'prod_names': ['mosaic'],    #['LAI', 'fCOVER', 'fAPAR', 'Albedo'], 
    'resolution': 20,            # Exporting spatial resolution    
    'out_folder': 'C:/Work_Data/S2_tile55_2022_10m_new1',  # the folder name for exporting
    'projection': 'EPSG:3979',
    'IncludeAngles': False,
    #'start_dates': ['2022-06-15'],
    #'end_dates': ['2022-09-15'],
    'regions': {'ottawa': ottawa_region}
}


if __name__ == "__main__":
    main()
