import os
import glob
import datetime
import rasterio
import rioxarray

import numpy as np
import xarray as xr

from pathlib import Path

import eoTileGrids as eoTG
import eoImage as eoIM






##############################################################################################################
# Description: This function loads historic EO bands from TIF files into an xarray Dataset
# 
#
##############################################################################################################  
def load_historic_bands(YearPaths:list[str], TileName:str, Month:str, Bands:list[str], FilePattern:str):
  """
    Inputs:
        inYearPaths:  List of paths to directories for each year
        TileName:     Name of the tile to load (e.g., 'tile55_422')
        Month:        Month to load (e.g., 'Aug', 'Sep' etc.)
        Bands:      List of band names to load (e.g., ['red', 'nor', 'swir16'])
        FilePattern:File naming pattern with placeholders for year and band (e.g., 'EO_{year}_{band}.tif')
    
    Outputs:
        xrDS:         xarray Dataset containing the loaded bands"""

  if not eoTG.valid_tile_name(TileName):
    print(f"Error: Invalid tile name {TileName}.")
    return None  
  
  #==========================================================================================================
  # Confirm all the given directories in "YearPaths" are valid
  #==========================================================================================================
  for year_path in YearPaths:
    if not os.path.isdir(year_path):
      print(f"Error: Year path {year_path} does not exist or is not a directory.")
      return None

  #==========================================================================================================
  # Search each year data directory for the specified tile
  #==========================================================================================================
  img_list = []
  for year_path in YearPaths:    
    subdirs = [p for p in Path(year_path).iterdir() if p.is_dir()]
    matched_dir = [p for p in subdirs if TileName in p.name]

    if len(matched_dir) != 1:
      print(f"Error: Too many subdirectories matched {TileName} in {year_path}.")
      return None

    temp_rxDS = eoIM.load_TIF_files_to_xr(matched_dir[0], Bands, Month)
    if temp_rxDS is not None:
      img_list.append(temp_rxDS)

  img_list = [da.to_dataset(dim='band') for da in img_list]

  # Combine datasets along a new dimension if needed (or merge them)
  outrxDS = xr.merge(img_list, compat='override')
  print(outrxDS)

  return outrxDS






YearPaths = ['E:\HLS_mosaics_2022', 'E:\HLS_mosaics_2023', 'E:\HLS_mosaics_2024', 'E:\HLS_mosaics_2025']
load_historic_bands(YearPaths, 'tile55_422', 'Aug', ['blue', 'nir'], '')
