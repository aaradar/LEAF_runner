# Production

## Line 234 - 248 (remove runner keys that are invalid in production)
  #==========================================================================================================
  #Remove the keys that are invalid in mosaic production
  #==========================================================================================================
  INVALID_KEYS = {
    "regions_start_index",
    "regions_end_index",
    "spatial_buffer_m",
    "temporal_buffer",
    "num_years",
    "file_variables"
  }

  for key in list(inProdParams.keys()):
      if key in INVALID_KEYS:
          inProdParams.pop(key
    
# eoMosiac
## Line 1225-1227
    # Always use load_STAC_items with the requested resolution to ensure all bands are loaded
    # in the same coordinate system. Let odc.stac.load() handle the resampling automatically.
    xrDS_S2 = load_STAC_items(filtered_items['S2'], Bands, ChunkDict, ProjStr, Scale)  

# eoParams

## Line 75 (to get fcover to work)
    before
    if 'lai' in prod_names or 'fcov' in prod_names or 'fap' in prod_names or 'alb' in prod_names:
    after
    if 'lai' in prod_names or 'fcover' in prod_names or 'fapar' in prod_names or 'albebo' in prod_names:


## Line 133-139 (to get 10m to work)
   if 'bands' not in inParams:
      # Always use ALL_BANDS for Sentinel-2 to ensure rededge bands are loaded for LEAF processing
      query_conds['bands'] = SsrData['ALL_BANDS'] + ['scl']

    else : 
      # Always use ALL_BANDS for Sentinel-2 to ensure rededge bands are loaded for LEAF processing
      required_bands = SsrData['ALL_BANDS'] + ['scl']

## Line 961 (different warning message)
    print('\n<get_time_window> one of required keys is not exist to form time.')