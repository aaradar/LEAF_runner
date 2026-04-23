import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import concurrent.futures
from datetime import datetime


import eoImage as eoIM
import eoMosaic as eoMz
import eoUtils as eoUs
import eoParams as eoPM
import eoTileGrids as eoTG
import eoAuxData as eoAD

import SL2P_V1
import SL2P_NetsTools





#############################################################################################################
# Description: This function produces all the required vegetation parameter maps for ONE tile and ONE month.
# 
# Note: This function is equivalent to the 'SL2P_estimation' in GEE LEAF package
#
# Revision history:  2024-May-24  Lixin Sun  Initial creation
#                    2024-Jul-20  Lixin Sun  Modified to generate the final composite image tile by tile.
#
#############################################################################################################
def create_LEAF_maps(ProdParams, CompParams):
  '''
    Args:
      inParams(dictionary): A dictionary containing all execution parameters.'''

  leaf_start = time.time()
  #==========================================================================================================
  # Validate input parameters
  #==========================================================================================================
  if len(ProdParams['current_region']) < 6: #Ensure the existence of a valid 'current_region' item
    print('<create_LEAF_maps> Invalid <current_tile> item in parameter dictionary!')
    return None
  
  #==========================================================================================================
  # Create the required mosaic image
  #==========================================================================================================
  prod_names = ProdParams['prod_names']
  #print(f'\n<create_LEAF_maps> all parameters for generating mosaic: {ProdParams}')

  mosaic = eoMz.one_mosaic(ProdParams, Output=False)  # REMOVED the third argument
  print('\n<create_LEAF_maps> The bands in mosaic image:', mosaic.data_vars)
  # print('\n<create_LEAF_maps> ext_tiffs_rec = ', ext_tiffs_rec)
  # print('\n<create_LEAF_maps> period_str = ', period_str)
  
  #return mosaic
  ProdParams['prod_names'] = prod_names
  #print(f'\n\n\n\n<create_LEAF_maps> all parameters after generating mosaic: {ProdParams}')
  #==========================================================================================================
  # Convert the angle data variables, 'VZA', 'VAA', 'SZA', and 'SAA', to three cos data variables
  #==========================================================================================================
  mosaic = mosaic.assign(cosSZA = np.cos(np.deg2rad(mosaic['SZA'])),
                         cosVZA = np.cos(np.deg2rad(mosaic['VZA'])),
                         cosRAA = np.cos(np.deg2rad(mosaic['SAA'] - mosaic['VAA'])))
  
  # Drop off 'VZA', 'VAA', 'SZA', and 'SAA' data variables
  mosaic = mosaic.drop_vars(['SZA', 'SAA', 'VZA', 'VAA'])
  print('\n<create_LEAF_maps> The bands in modified mosaic image:', mosaic.data_vars)

  #==========================================================================================================
  # (1) Read and clip land cover map based on the spatial extent of "entire_map"
  # (2) Create a network ID map with the same spatial dimensions as clipped landcover map
  #==========================================================================================================  
  sub_LC_map = eoAD.get_local_CanLC('F:\\Canada_LC_maps\\Canada_LC_2025_30m.tif', mosaic) # for workstation at Observatory
  #sub_LC_map = eoAD.get_local_CanLC('C:\\Work_Data\\Canada_LC_maps\\Canada_LC_2020_30m.tif', mosaic) # for work laptop
  SsrData    = eoIM.SSR_META_DICT[str(ProdParams['sensor']).upper()]

  DS_Options = SL2P_V1.make_DS_options('sl2p_nets', SsrData)  
  netID_map  = SL2P_NetsTools.makeIndexLayer(sub_LC_map, DS_Options)

  #==========================================================================================================
  # Define a function that can produce vegetation parameter maps for ONE granule
  #==========================================================================================================
  print('\n<create_LEAF_maps> Bands in mosaic before rescaling:', list(mosaic.data_vars))
  ready_mosaic = eoIM.rescale_spec_bands(mosaic, SsrData['LEAF_BANDS'], 0.01, 0)
  print('<create_LEAF_maps> Bands in ready_mosaic after rescaling:', list(ready_mosaic.data_vars))
  out_VP_maps  = SL2P_NetsTools.estimate_VParams(ProdParams, DS_Options, ready_mosaic, netID_map)  
    
  #==========================================================================================================
  # Display the elapsed time for entire process
  #==========================================================================================================
  leaf_stop = time.time()
  leaf_time = (leaf_stop - leaf_start)/60
  print(f'\n\n<<< The elapsed time for generating one monthly tile product = {leaf_time} minutes>>>')

  return out_VP_maps




#############################################################################################################
# Description: Produces LEAF products for a customized spatial region and time window or a specified scene
# 
# Note: This function will be called when one of the following three situations happens:
#       (1) A ee.Geometry.Polygon object is provided as the value corresponding to "custom_region" key
#       (2) A user-specified scene ID is provided as the value corresponding to "scene_ID" key
#       (3) A time window is provided as the values corresponding to "start_date" and "end_date" keys
#
# Revision history:  2023-Nov-26  Lixin Sun  Initial creation 
#
#############################################################################################################
def SL2P_estimation(Params):
  '''Produces LEAF products for one or multiple tiles in CANADA

    Args:
      Params(Dictionary): A dictionary containing all execution input parameters.'''  
  
  #==========================================================================================================
  # Obtain some required parameters
  #==========================================================================================================
  SsrData     = eoIM.SSR_META_DICT[Params['sensor']]
  year        = int(Params['year'])
  SceneID     = str(Params['scene_ID'])    # An optional ID of a single scene/granule 
  ProductList = Params['prod_names']       # A list of products to be generated
  tile_name   = str(Params['current_tile'])

  #==========================================================================================================
  # Obtain timeframe and spatial region
  #==========================================================================================================
  start, stop = eoPM.get_time_window(Params)
  region      = eoPM.get_spatial_region(Params)
  if len(tile_name) > 2:
     region = eoTG.expandSquare(region, 0.02)  

  #print('<apply_SL2P> All parameters:', Params) 

  #==========================================================================================================
  # Obtain a global Land cover classification map and export it as needed 
  #==========================================================================================================
  ClassImg = eoAD.get_GlobLC(year, False).uint8().clip(region)

  #==========================================================================================================
  # If scene_ID is provided, ontain its footprint as ROI
  #==========================================================================================================
  if len(SceneID) > 5: 
    # Obtain the specified single scene and its footprint
    ssr_code, tile_str, refer_date_str, valid_ID = eoIM.parse_ImgID(SceneID)  # parse the given image ID string
    '''
    if valid_ID == True and SsrData['SSR_CODE'] == ssr_code:
      image  = ee.Image(SsrData['GEE_NAME'] + '/' + SceneID) 
      image  = eoIM.apply_gain_offset(image, SsrData, 1, False)  # convert SR to range between 0 and 1
      image  = eoIM.attach_AngleBands(image, SsrData)            # attach three imaging angle bands
      region = ee.Image(image).geometry()
      
      SL2P_separate_params(Params, image, region, SsrData, ClassImg)
    '''
  else: 
    mosaic = eoMz.LEAF_Mosaic(SsrData, region, start, stop, True)   
    print("<apply_SL2P> The band names in mosiac image = ", mosaic.bandNames().getInfo())

    SL2P_separate_params(Params, mosaic, region, SsrData, ClassImg)

    if Is_export_required('date', ProductList):
      Params['prod_name'] = 'Date'
      date_map = mosaic.select([eoIM.pix_date])
      export_one_map(Params, region, date_map, 'Date')        




# ─────────────────────────────────────────────────────────────────────────────
# _write_VP_csv 
# ─────────────────────────────────────────────────────────────────────────────
def _write_VP_csv(inParams, inXrDS, VP_scalers):
    import pandas as pd
    import pyproj

    # ── Build the path of the mosaic CSV that export_csv() already wrote ──
    SsrData      = eoIM.SSR_META_DICT[str(inParams['sensor'])]
    region_label = inParams.get('_region_label', str(inParams.get('current_region', 'region')))
    tile_label   = inParams.get('_tile_label', '')
    period_str   = str(inParams.get('time_str', 'period'))
    spa_scale    = inParams.get('csv_scale', inParams.get('resolution', 30))

    if tile_label:
        filename = f"{SsrData['NAME']}_{region_label}_{tile_label}_{period_str}_{spa_scale}m.csv"
    else:
        filename = f"{SsrData['NAME']}_{region_label}_{period_str}_{spa_scale}m.csv"

    csv_path = os.path.join(inParams['out_folder'], filename)

    if not os.path.exists(csv_path):
        print(f'<_write_VP_csv> WARNING: mosaic CSV not found at {csv_path}. '
              f'Run export_csv() first. Writing standalone VP CSV as fallback.')
        # ── Fallback: write a VP-only CSV with a distinct suffix ──────────
        if '_clip_geom' in inParams:
            mosaic_to_sample = eoMz._clip_mosaic_to_regions(inParams, inXrDS)
        else:
            mosaic_to_sample = inXrDS
        skip_bands  = {'spatial_ref'}
        band_names  = [v for v in mosaic_to_sample.data_vars if v not in skip_bands]
        stacked     = mosaic_to_sample[band_names].stack(pixel=('y', 'x'))
        pixel_index = stacked.coords['pixel'].values
        df          = pd.DataFrame({b: stacked[b].values for b in band_names})
        if inParams.get('csv_dropNulls', True):
            mask        = df.notna().all(axis=1)
            df          = df[mask].reset_index(drop=True)
            pixel_index = pixel_index[mask.values if hasattr(mask, 'values') else mask]
        proj_str    = inParams.get('projection', 'EPSG:3979')
        transformer = pyproj.Transformer.from_crs(pyproj.CRS.from_string(proj_str),
                                                   pyproj.CRS.from_epsg(4326), always_xy=True)
        ys, xs      = np.array([p[0] for p in pixel_index]), np.array([p[1] for p in pixel_index])
        lons, lats  = transformer.transform(xs, ys)
        df.insert(0, 'latitude', lats)
        df.insert(0, 'longitude', lons)
        for b in band_names:
            if b in df.columns:
                df[b] = df[b] * VP_scalers.get(b, 1)
        fallback_path = csv_path.replace('.csv', '_VP_only.csv')
        os.makedirs(inParams['out_folder'], exist_ok=True)
        df.to_csv(fallback_path, index=False)
        print(f'<_write_VP_csv> Standalone VP CSV → {fallback_path}')
        return

    # ── Load the existing mosaic CSV ──────────────────────────────────────
    mosaic_df = pd.read_csv(csv_path)
    print(f'<_write_VP_csv> Loaded mosaic CSV ({len(mosaic_df):,} rows) from {csv_path}')

    # ── Clip, stack, and build a VP DataFrame ────────────────────────────
    if '_clip_geom' in inParams:
        vp_clipped = eoMz._clip_mosaic_to_regions(inParams, inXrDS)
    else:
        vp_clipped = inXrDS

    skip_bands    = {'spatial_ref'}
    vp_band_names = [v for v in vp_clipped.data_vars if v not in skip_bands]
    stacked       = vp_clipped[vp_band_names].stack(pixel=('y', 'x'))
    pixel_index   = stacked.coords['pixel'].values
    vp_df         = pd.DataFrame({b: stacked[b].values for b in vp_band_names})

    valid_mask  = vp_df.notna().any(axis=1)
    vp_df       = vp_df[valid_mask].reset_index(drop=True)
    pixel_index = pixel_index[valid_mask.values if hasattr(valid_mask, 'values') else valid_mask]
    print(f'<_write_VP_csv> VP pixels after null filter: {len(vp_df):,}')

    proj_str    = inParams.get('projection', 'EPSG:3979')
    transformer = pyproj.Transformer.from_crs(pyproj.CRS.from_string(proj_str),
                                               pyproj.CRS.from_epsg(4326), always_xy=True)
    ys, xs      = np.array([p[0] for p in pixel_index]), np.array([p[1] for p in pixel_index])
    lons, lats  = transformer.transform(xs, ys)
    vp_df.insert(0, 'latitude',  lats)
    vp_df.insert(0, 'longitude', lons)

    for b in vp_band_names:
        if b in vp_df.columns:
            vp_df[b] = vp_df[b] * VP_scalers.get(b, 1)

    # ── Left-join VP columns onto the mosaic rows via rounded lat/lon key ─
    DECIMALS = 6
    mosaic_df['_lat_key'] = mosaic_df['latitude'].round(DECIMALS)
    mosaic_df['_lon_key'] = mosaic_df['longitude'].round(DECIMALS)
    vp_df['_lat_key']     = vp_df['latitude'].round(DECIMALS)
    vp_df['_lon_key']     = vp_df['longitude'].round(DECIMALS)

    combined_df = mosaic_df.merge(
        vp_df[['_lat_key', '_lon_key'] + vp_band_names],
        on  = ['_lat_key', '_lon_key'],
        how = 'left',
    ).drop(columns=['_lat_key', '_lon_key'])

        # Use the first VP band that actually landed in combined_df as the match counter
    _check_band = next((b for b in vp_band_names if b in combined_df.columns), None)
    if _check_band:
        matched = combined_df[_check_band].notna().sum()
        print(f'<_write_VP_csv> {matched:,} / {len(combined_df):,} rows matched VP values (checked via {_check_band!r}).')
    else:
        print('<_write_VP_csv> WARNING: no VP bands found in combined_df after merge.')

    # ── Overwrite the CSV in-place with the combined result ───────────────
    combined_df.to_csv(csv_path, index=False)
    print(f'<_write_VP_csv> Updated CSV → {csv_path}  '
          f'({len(combined_df):,} rows, {len(combined_df.columns)} columns)')


# ─────────────────────────────────────────────────────────────────────────────
# _write_VP_geotiffs  — unchanged from original, included for completeness
# ─────────────────────────────────────────────────────────────────────────────
def _write_VP_geotiffs(inParams, inXrDS, VP_scalers):
    """Write per-band GeoTIFFs for one region (after clipping if needed)."""
    # Clip if a geometry was injected (tiles mode), otherwise use full dataset
    if '_clip_geom' in inParams:
        mosaic_to_write = eoMz._clip_mosaic_to_regions(inParams, inXrDS)
    else:
        mosaic_to_write = inXrDS
 
    rio_xrDS = mosaic_to_write.rio.write_crs(inParams['projection'], inplace=True)
 
    dir_path = inParams['out_folder']
    os.makedirs(dir_path, exist_ok=True)
 
    SsrData      = eoIM.SSR_META_DICT[str(inParams['sensor'])]
    region_label = inParams.get('_region_label', str(inParams.get('current_region', 'region')))
    tile_label   = inParams.get('_tile_label', '')
    period_str   = str(inParams['time_str'])
    spa_scale    = inParams['resolution']
    export_style = str(inParams['export_style']).lower()
 
    if tile_label:
        filePrefix = f"{SsrData['NAME']}_{region_label}_{tile_label}_{period_str}"
    else:
        filePrefix = f"{SsrData['NAME']}_{region_label}_{period_str}"
 
    if 'sepa' in export_style:
        for band in rio_xrDS.data_vars:
            out_img     = (rio_xrDS[band] * VP_scalers.get(band, 1)).astype(np.uint8)
            filename    = f"{filePrefix}_{band}_{spa_scale}m.tif"
            output_path = os.path.join(dir_path, filename)
            out_img.rio.to_raster(output_path)
            print(f'<_write_VP_geotiffs> Wrote {output_path}')
    else:
        filename    = f"{filePrefix}_LEAF_{spa_scale}m.tif"
        output_path = os.path.join(dir_path, filename)
        rio_xrDS.to_netcdf(output_path)
        print(f'<_write_VP_geotiffs> Wrote {output_path}')
 
 
# ─────────────────────────────────────────────────────────────────────────────
# export_VegParamMaps  — updated orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def export_VegParamMaps(inParams, inXrDS):
    """
    Export vegetation-parameter maps as GeoTIFFs or a CSV file.
 
    Reads inParams['output_type'] ('geotiff' by default, or 'csv') and
    dispatches to the appropriate writer.  Both 'regions' mode and 'tiles'
    mode are supported, matching the behaviour of eoMosaic.export_mosaic().
 
    tiles mode
    ----------
    Calls eoMosaic._get_tile_regions() to find all regions that intersect
    the current tile and match the current time window, then calls the writer
    once per region with '_clip_geom', '_region_label', and '_tile_label'
    injected into a shallow copy of inParams.  If no matching regions are
    found the export is skipped entirely.
 
    regions mode  (default)
    -----------------------
    Calls the writer once with inParams unchanged; clipping is handled
    inside the writer via eoMosaic._clip_mosaic_to_regions().
 
    Parameters
    ----------
    inParams : dict
        Parameter dictionary.  Relevant keys: 'output_type', 'mode',
        'current_region', 'sensor', 'projection', 'out_folder', 'time_str',
        'resolution', 'export_style', plus all keys required by the chosen
        writer.
    inXrDS : xarray.Dataset
        Vegetation-parameter dataset produced by create_LEAF_maps().
    """
    print('\n\n<export_VegParamMaps> data variables in given VP map:', inXrDS.data_vars)
 
    # ── Build VP scale-factor dict (same logic as before) ─────────────────
    VP_scalers = {}
    for s in inXrDS.data_vars:
        S = s.upper()
        if 'LAI' in S:
            VPOptions = SL2P_V1.make_VP_options('lai')
            VP_scalers[s] = VPOptions['scale_factor']
        elif 'FAPAR' in S or 'FCOVER' in S or 'ALBEDO' in S:
            VPOptions = SL2P_V1.make_VP_options('FAPAR')
            VP_scalers[s] = VPOptions['scale_factor']
        else:
            VP_scalers[s] = 1
 
    # ── Choose writer based on output_type ────────────────────────────────
    output_type = str(inParams.get('output_type', 'geotiff')).lower()
    writer      = _write_VP_csv if output_type == 'csv' else _write_VP_geotiffs
 
    # ── Dispatch: tiles vs regions mode ───────────────────────────────────
    mode = str(inParams.get('mode', 'regions')).lower()
 
    if mode == 'tiles':
        # Find regions that spatially and temporally match this tile
        tile_regions = eoMz._get_tile_regions(inParams, inXrDS)
 
        if not tile_regions:
            print(
                '<export_VegParamMaps> tiles mode: no matching regions for '
                'this tile/time window — skipping.'
            )
            return
 
        tile_label = str(inParams.get('current_region', ''))
 
        for reg_name, proj_geom in tile_regions.items():
            print(
                f'\n<export_VegParamMaps> tiles mode — '
                f'exporting region: {reg_name} (tile: {tile_label})'
            )
            region_params = {
                **inParams,
                '_clip_geom':    proj_geom,
                '_region_label': reg_name,
                '_tile_label':   tile_label,
            }
            writer(region_params, inXrDS, VP_scalers)
 
    else:  # 'regions' mode (default)
        writer(inParams, inXrDS, VP_scalers)
 


#############################################################################################################
# Description: This fuction produces vegetation biophysical parameter maps according to given parameters.
# 
# Revision history:  2024-Jul-30  Lixin Sun  Initial creation
#
#############################################################################################################
def LEAF_production(ProdParams, CompParams):
  '''Produces vegetation biophysical parameter maps according to given parameters.

     Args:
       ProdParams(Python Dictionary): A dictionary containing input parameters related to data production;
       CompParams(Python Dictionary): A dictionary containing input parameters related to the computing environment.
  '''

  #==========================================================================================================
  # When region-specific dates are provided but no global months/start_dates exist, inject a placeholder
  # so that standardize_params never returns None/None. Real per-region dates overwrite before any call.
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
    first_region      = next(iter(ProdParams['region_start_dates']))
    placeholder_start = ProdParams['region_start_dates'][first_region][0]
    placeholder_end   = ProdParams['region_end_dates'].get(first_region, [placeholder_start])[0]
    ProdParams['start_dates'] = [placeholder_start]
    ProdParams['end_dates']   = [placeholder_end]
    ProdParams['monthly']     = False
    print(f'<LEAF_production> Injected placeholder dates for standardization: '
          f'{placeholder_start} / {placeholder_end}')

  #==========================================================================================================
  # Standardize the input parameters — tile_names must remain intact for this to succeed
  #==========================================================================================================
  usedParams = eoPM.get_LEAF_params(ProdParams, CompParams)
  print('<LEAF_production> All input parameters = ', usedParams)

  #==========================================================================================================
  # Produce vegetation biophysical parameter maps for each region and time window
  #==========================================================================================================
  region_names = list(usedParams['regions'].keys())

  # Check for region-specific dates
  has_region_dates = (
      'region_start_dates' in usedParams and
      'region_end_dates'   in usedParams and
      len(usedParams['region_start_dates']) > 0
  )

  # If region-specific dates are driving this run, never fall back to default/placeholder
  # dates for regions not explicitly listed. Only use defaults in true global-dates mode.
  default_start_dates = [] if has_region_dates else usedParams.get('start_dates', [])
  default_end_dates   = [] if has_region_dates else usedParams.get('end_dates',   [])

  for reg_name in region_names:
    usedParams = eoPM.set_spatial_region(usedParams, reg_name)

    # Check if region has region-specific dates
    if has_region_dates and reg_name in usedParams['region_start_dates']:
      # Get region-specific dates
      region_start_dates = usedParams['region_start_dates'][reg_name]
      region_end_dates   = usedParams['region_end_dates'].get(reg_name, region_start_dates)

      # Use region-specific dates
      print(f'\n<LEAF_production> Using region-specific dates for {reg_name}')
      print(f'  Start dates: {region_start_dates}')
      print(f'  End dates:   {region_end_dates}')

      usedParams['start_dates'] = region_start_dates
      usedParams['end_dates']   = region_end_dates
      usedParams['monthly']     = False
      nTimes = len(region_start_dates)

    elif default_start_dates and default_end_dates:
      # Use default/global dates — only reachable when region_start_dates was never set
      print(f'\n<LEAF_production> Using global dates for {reg_name}')
      usedParams['start_dates'] = default_start_dates
      usedParams['end_dates']   = default_end_dates
      nTimes = len(default_start_dates)

    else:
      # Tile-derived regions and any other region without explicit dates get skipped cleanly
      print(f'\n<LEAF_production> WARNING: Region {reg_name} has no dates - SKIPPING')
      continue

    # Process all time windows for this region
    for TIndex in range(nTimes):
      sedParams = eoPM.set_current_time(usedParams, TIndex)

      # ------------------------------------------------------------------
      # Guard: set_current_time updates Criteria['timeframe'], but only
      # when get_time_window succeeds.
      # Double-check here so that a bad timeframe string never reaches
      # the STAC API.
      # ------------------------------------------------------------------
      start_check = usedParams['start_dates'][TIndex]
      end_check   = usedParams['end_dates'][TIndex]

      if start_check is None or end_check is None:
          print(
              f"<MosaicProduction> WARNING: None date for {reg_name} "
              f"at index {TIndex}, skipping."
          )
          continue

      # Explicitly refresh Criteria timeframe in case set_current_time
      # silently failed to update it (e.g. due to a previous None state).
      if 'Criteria' in usedParams:
          usedParams['Criteria']['timeframe'] = f"{start_check}/{end_check}"
          usedParams['Criteria']['region'] = usedParams['regions'][reg_name]

      print(
          f"<MosaicProduction> Processing {reg_name} | "
          f"time window: {start_check} / {end_check}"
      )
      # Produce and export vegetation biophysical parameter maps
      out_style = str(usedParams['export_style']).lower()
      if 'comp' in out_style:
        print('\n<LEAF_production> Generate and export biophysical maps in one file .......')
        # compact export not yet implemented
      else:
        print('\n<LEAF_production> Generate and export separate vegetation biophysical maps......')
        VBP_maps = create_LEAF_maps(usedParams, CompParams)
        export_VegParamMaps(usedParams, VBP_maps)

  # Restore only real global dates — placeholder must not leak out of this function
  if default_start_dates:
    usedParams['start_dates'] = default_start_dates
    usedParams['end_dates']   = default_end_dates