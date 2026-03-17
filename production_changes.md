# Production
# Production.py 
## Line 110 - 135

  #==========================================================================================================
  #When region-specific dates are provided but no global months/start_dates exist, inject a placeholder
  #Start/end date so that standardize_params -> form_time_windows -> get_time_window never returns None/None.
  #The real per-region dates will overwrite these before any mosaic call is made.
  #==========================================================================================================
  has_region_dates_input = (
      'region_start_dates' in ProdParams and
      'region_end_dates'   in ProdParams and
      len(ProdParams['region_start_dates']) > 0
  )
  no_global_dates = (
      not ProdParams.get('months') and
      not ProdParams.get('start_dates')
  )

  if has_region_dates_input and no_global_dates:
    # Pick the first available date from any region as a safe placeholder for standardization.
    # The real per-region dates are applied inside the loop below, before one_mosaic() is called.
    first_region      = next(iter(ProdParams['region_start_dates']))
    placeholder_start = ProdParams['region_start_dates'][first_region][0]
    placeholder_end   = ProdParams['region_end_dates'].get(first_region, [placeholder_start])[0]
    ProdParams['start_dates'] = [placeholder_start]
    ProdParams['end_dates']   = [placeholder_end]
    ProdParams['monthly']     = False
    print(f'<MosaicProduction> Injected placeholder dates for standardization: '
          f'{placeholder_start} / {placeholder_end}')

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

    
# LEAFProduction.py

## def LEAF_production(ProdParams, CompParams):
    To work with region dates

## Line 52 (to not crash)
     mosaic = eoMz.one_mosaic(ProdParams, CompParams)  # ← REMOVED the third argument

# SL2P_NetTools.py 

## Line 140 - 141 (to make LC make work with sizes other than 30m)
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
  

# eoMosiac.py 

## Line 1225-1227
    # Always use load_STAC_items with the requested resolution to ensure all bands are loaded
    # in the same coordinate system. Let odc.stac.load() handle the resampling automatically.
    xrDS_S2 = load_STAC_items(filtered_items['S2'], Bands, ChunkDict, ProjStr, Scale)  


##  Rewrote search stac catalog to work with new code
def search_STAC_Catalog(inParams, MaxImgs):
  '''
    Args:
      inParams(dictionary): A dictionary containing all parameters for generating a composite image;
      MaxImgs(int): A specified maximum number of images in a queried list.
  '''
  
  #==========================================================================================================
  # Use publicly available STAC 
  #==========================================================================================================
  Criteria = inParams['Criteria']

  #==========================================================================================================
  # Fix winding order: NASA CMR STAC requires counter-clockwise winding order (right-hand rule).
  # Reorder polygon coordinates using shapely to ensure compliance.
  #==========================================================================================================
  from shapely.geometry import shape, mapping
  from shapely.geometry.polygon import orient
  Region = Criteria['region']
  Region = mapping(orient(shape(Region), sign=1.0))
  Criteria['region'] = Region
  print('<search_STAC_Images> The given region = ', Region)

  #==========================================================================================================
  # Fix datetime format: NASA CMR requires full RFC3339 timestamps with time and Z suffix.
  # e.g. '2025-08-01T00:00:00Z/2025-08-31T23:59:59Z' instead of '2025-08-01/2025-08-31'
  #==========================================================================================================
  tf     = str(Criteria['timeframe'])
  parts  = tf.split('/')
  start_dt = parts[0] if 'T' in parts[0] else parts[0] + 'T00:00:00Z'
  end_dt   = parts[1] if 'T' in parts[1] else parts[1] + 'T23:59:59Z'
  datetime_str = f'{start_dt}/{end_dt}'

  is_nasa = 'earthdata.nasa.gov' in str(Criteria['catalog']) or 'LPCLOUD' in str(Criteria['catalog'])

  if is_nasa:
    #========================================================================================================
    # For NASA LPCLOUD: bypass pystac_client entirely to avoid the CMR GraphQL "context creation" error.
    # Instead, POST directly to the STAC search endpoint using an earthaccess authenticated session.
    # This also avoids the cloud_cover query filter which NASA's STAC does not support reliably.
    # Limit is capped at 200 to stay safely under the 255-item AWS Lambda 6MB payload limit.
    #========================================================================================================
    import json
    import pystac
    import earthaccess

    earthaccess.login(strategy="netrc")
    session    = earthaccess.get_requests_https_session()
    search_url = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search"
    stac_items = []

    for coll in Criteria['collection']:
      payload = {
        "collections": [coll],
        "intersects":  Region,
        "datetime":    datetime_str,
        "limit":       min(MaxImgs, 200)   # stay under the 255-item Lambda payload cap
      }
      response = session.post(search_url, json=payload)
      response.raise_for_status()
      features = response.json().get("features", [])
      print(f"<search_STAC_Catalog> Number of items found in collection {coll}: {len(features)}")

      if not features:
        print(f"<search_STAC_Catalog> WARNING: No items found in {coll}, skipping.")
        continue

      for f in features:
        stac_items.append(pystac.Item.from_dict(f))

    if not stac_items:
      raise ValueError("No STAC items found. Adjust region, time range, or filters.")

  else:
    #========================================================================================================
    # For AWS Element84 (Sentinel-2, Landsat): use pystac_client normally.
    # The cloud_cover query filter is supported here.
    #========================================================================================================
    catalog    = psc.Client.open(str(Criteria['catalog']))
    stac_items = []

    nCollections = len(Criteria['collection'])
    if nCollections > 1:
      for coll in Criteria['collection']:
        stac_catalog = catalog.search(collections = [coll],
                                      intersects  = Region,
                                      datetime    = datetime_str,
                                      query       = Criteria['filters'],
                                      limit       = MaxImgs)

        items = list(stac_catalog.items())
        print(f"<search_STAC_Catalog> Number of items found in collection {coll}: {len(items)}")

        if not items:
          raise ValueError("No STAC items found. Adjust region, time range, or filters.")

        stac_items.extend(items)

    else:
      stac_catalog = catalog.search(collections = Criteria['collection'],
                                    intersects  = Region,
                                    datetime    = datetime_str,
                                    query       = Criteria['filters'],
                                    limit       = MaxImgs)

      stac_items = list(stac_catalog.items())

  #==========================================================================================================
  # Ingest imaging geometry angles into each STAC item
  # Note: only needed for AWS data catalog, where angles are not embedded in item properties
  #==========================================================================================================
  ssr_str = str(inParams['sensor']).lower()
  if 'hls' not in ssr_str:
    stac_items, angle_time = ingest_Geo_Angles(stac_items)
    print('\n<search_STAC_Catalog> The total elapsed time for ingesting angles = %6.2f minutes'%(angle_time))

  return stac_items

# eoParams.py 

## def form_time_windows(inParams):

  if not has_custom_window(inParams):
    months = inParams.get('months', [])

    if months:
      # Standard monthly workflow: build start/end date lists from the month list
      inParams['monthly'] = True
      year = inParams['year']
      for index, month in enumerate(months):
        start, end = eoUs.month_range(year, month)
        if index == 0:
          inParams['start_dates'] = [start]
          inParams['end_dates']   = [end]
        else:  
          inParams['start_dates'].append(start)
          inParams['end_dates'].append(end)

    else:
      # No months specified and no custom window — this is expected when the caller
      # is using the region-specific dates workflow and has already injected placeholder
      # start/end dates.  Do NOT overwrite them; just ensure monthly=False.
      if not inParams.get('start_dates'):
        print('<form_time_windows> WARNING: No months and no start_dates defined. '
              'start_dates will be empty — ensure region-specific dates are injected '
              'before calling one_mosaic().')
      inParams['monthly'] = False

  elif 'standardized' not in inParams:
    inParams['monthly'] = False
  
  inParams['current_time'] = 0

  return inParams

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