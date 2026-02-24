# Production
## Line 144 - 211 (to process specific region dates)
    else:
    region_names = list(usedParams['regions'].keys())
    
    # Check for region-specific dates
    has_region_dates = (
        'region_start_dates' in usedParams and 
        'region_end_dates' in usedParams and
        len(usedParams['region_start_dates']) > 0
    )
    
    # Store default dates
    default_start_dates = usedParams.get('start_dates', [])
    default_end_dates = usedParams.get('end_dates', [])

    for reg_name in region_names:
        usedParams = eoPM.set_spatial_region(usedParams, reg_name)
        
        # Check if region has region-specific dates
        if has_region_dates and reg_name in usedParams['region_start_dates']:
            # Get region-specific dates
            region_start_dates = usedParams['region_start_dates'][reg_name]
            region_end_dates = usedParams['region_end_dates'].get(reg_name, region_start_dates)
            
            # Validate dates (check if any date is before 2016 for Sentinel-2)
            valid_dates = True
            for date in region_start_dates + region_end_dates:
                if isinstance(date, str):
                    date_year = int(date.split('-')[0])
                else:
                    date_year = date.year
                
                if date_year < 2016:
                    print(f'\n<MosaicProduction> WARNING: Region {reg_name} has dates before 2016 (Sentinel-2 launch)')
                    print(f'  Region dates: {region_start_dates} to {region_end_dates}')
                    print(f'  SKIPPING this region')
                    valid_dates = False
                    break
            
            if not valid_dates:
                continue  # Skip this region
            
            # Use region-specific dates
            print(f'\n<MosaicProduction> Using region-specific dates for {reg_name}')
            print(f'  Start dates: {region_start_dates}')
            print(f'  End dates: {region_end_dates}')
            
            usedParams['start_dates'] = region_start_dates
            usedParams['end_dates'] = region_end_dates
            nTimes = len(region_start_dates)
            
        else:
            # No region-specific dates found - skip this region
            print(f'\n<MosaicProduction> WARNING: Region {reg_name} has no region-specific dates')
            print(f'  SKIPPING this region')
            continue
        
        # Process all time windows for this region
        # Process all time windows for this region
        for TIndex in range(nTimes):
            # Set monthly flag based on whether we're using region dates
            usedParams['monthly'] = False  # Region-specific dates are not monthly
            usedParams = eoPM.set_current_time(usedParams, TIndex)
            
            # Process mosaic
            eoMz.one_mosaic(usedParams, CompParams)
    # NEW: Restore default dates after processing
    usedParams['start_dates'] = default_start_dates
    usedParams['end_dates'] = default_end_dates

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
    
# LEAFProduction
## def LEAF_production(ProdParams, CompParams):
    To work with region dates

## Line 52 (to not crash)
     mosaic = eoMz.one_mosaic(ProdParams, CompParams)  # ← REMOVED the third argument

# SL2P_NetTools (to make LC make work with sizes other than 30m)
## Line 140 - 141 
  #Fill NaN values with 0, then cast to int
  LCMap = LCMap.fillna(0).astype(int)

## Line 156 - 164
    # Find any unmapped LC values and warn user
  unique_lc_values = np.unique(LCMap.values).astype(int)
  unmapped_values = [v for v in unique_lc_values if v not in mapping_dict and not np.isnan(v)]
  if unmapped_values:
    warnings.warn(f"Land cover values {unmapped_values} not found in mapping dictionary. "
                  f"Mapping these to network ID 0. Mapped LC classes: {list(mapping_dict.keys())}")

  #Apply the mapping to the land cover map with default value 0 for unmapped classes
  netID_map_np = np.vectorize(lambda x: mapping_dict.get(x, 0), otypes=[int])(LCMap)
  

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