


######################################################################################################
# Description: This function creates a image collection acquired by a sensor over a geographical 
#              region during a period of time.
#
# Revision history:  2021-May-20  Lixin Sun  Updated with newly developed function "GEE_catalog_name" 
#                    2021-Jun-09  Lixin Sun  Modified by using "get_cloud_rate" to deteremine cloud
#                                            coverage percentage.
#                    2022-Nov-10  Lixin Sun  Added a default input parameter called "CloudRate", which 
#                                            is an optional cloud coverage percentage/rate.  
#                    2023-Apr-14  Lixin Sun  Added a case for "MODIS/061/MOD09A1", which does have any
#                                            image property
#                    2023-Sep-30  Lixin Sun  For Landsat 8 or 9, if target year is after 2022, then
#                                            both of them will be put into the returned collection.
#                    2023-Nov-09  Lixin Sun  Attach a "Cloud Score+" band to each image in a 
#                                            Sentinel-2 image collection.
######################################################################################################
#def getCollection(SsrData, Region, StartDate, EndDate, ExtraBandCode, CloudRate = -100):  
  '''Returns a image collection acquired by a sensor over a geographical region during a period of time  

  Arg: 
     SsrData(Dictionary): A Dictionary containing metadata associated with a sensor and data unit;
     Region(ee.Geometry): A geospatial polygon of ROI;
     StartDate(string or ee.Date): The start acquisition date string (e.g., '2020-07-01');
     EndDate(string or ee.Date): The stop acquisition date string (e.g., '2020-07-31');
     CloudRate(float): a given cloud coverage rate.'''
  '''  
  print('<getCollection> SsrData info:', SsrData)
  # Cast the input parameters into proper formats  
  region = ee.Geometry(Region)
  start  = ee.Date(StartDate)
  end    = ee.Date(EndDate)
  year   = int(start.get('year').getInfo())
  print('\n<getCollection> The year of time window = ', year) 

  #===================================================================================================
  # Determine a cloud coverage percentage/rate 
  # Note: there are two ways to determine a cloud coverage percebtage/rate
  # (1) based on sensor type and the centre of the given spatial region
  # (2) a given cloud coverage percentage/rate (CloudRate) 
  #===================================================================================================
  cloud_rate = Img.get_cloud_rate(SsrData, region) if CloudRate < 0 or CloudRate > 99.99 else ee.Number(CloudRate)
  print('<getCollection> Used cloud rate = ', cloud_rate.getInfo())

  #===================================================================================================
  # "filterMetadata" Has been deprecated. But tried to use "ee.Filter.gte(property, value)", did 
  # not work neither.
  #===================================================================================================
  CollName  = SsrData['GEE_NAME']  
  ssr_code  = SsrData['SSR_CODE']
  data_unit = SsrData['DATA_UNIT'] 

  if ssr_code == Img.MOD_sensor: # for MODIS data
    coll = ee.ImageCollection(CollName).filterBounds(region).filterDate(start, end) 
  elif ssr_code > Img.MAX_LS_CODE and ssr_code < Img.MOD_sensor: 
    # for Sentinel-2 data   
    # Note: Limiting SZA < 70.0 could lead to an empty image coolection for some Canadian Northen regions 
    coll = ee.ImageCollection(CollName).filterBounds(region).filterDate(start, end) \
               .filterMetadata(SsrData['CLOUD'], 'less_than', cloud_rate) \
               .filterMetadata('system:asset_size', 'greater_than', 1000000) # Added on Feb 13, 2024
               #.filterMetadata(SsrData['SZA'], 'less_than', 70.0) \
               #.limit(10000)
               #.filterMetadata(SsrData['CLOUD'], 'less_than', cloud_up) \
               
    #-------------------------------------------------------------------------------------------
    # Attach a "Cloud Score+" band to each image in Sentinel-2 image collection
    # Note: This function is unstable for now (Feb. 10, 2024)
    #-------------------------------------------------------------------------------------------
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED') \
               .filterBounds(region).filterDate(start, end)
    
    coll   = coll.map(lambda img: img.linkCollection(csPlus, ['cs']))

  elif ssr_code < Img.MAX_LS_CODE: 
    # for Landsat data
    if year < 2022 or ssr_code < Img.LS_sensor:
      # For one single Landsat sensor
      coll = ee.ImageCollection(CollName).filterBounds(region).filterDate(start, end).filterMetadata(SsrData['CLOUD'], 'less_than', cloud_rate) 

      if ExtraBandCode == Img.EXTRA_ANGLE and data_unit == Img.sur_ref: 
        toa_ssr_data = Img.SSR_META_DICT['L8_TOA']
        toa_coll     = ee.ImageCollection(toa_ssr_data['GEE_NAME']).filterBounds(region).filterDate(start, end).filterMetadata(toa_ssr_data['CLOUD'], 'less_than', cloud_rate) 
        
        coll = coll.map(lambda img: img.linkCollection(toa_coll, ['SZA', 'SAA', 'VZA', 'VAA']))

    else:
      if data_unit == Img.sur_ref:
        L8_sr_ssr_data = Img.SSR_META_DICT['L8_SR']
        L9_sr_ssr_data = Img.SSR_META_DICT['L9_SR']

        L8_sr_coll     = ee.ImageCollection(L8_sr_ssr_data['GEE_NAME']).filterBounds(region).filterDate(start, end).filterMetadata(L8_sr_ssr_data['CLOUD'], 'less_than', cloud_rate)         
        L9_sr_coll     = ee.ImageCollection(L9_sr_ssr_data['GEE_NAME']).filterBounds(region).filterDate(start, end).filterMetadata(L9_sr_ssr_data['CLOUD'], 'less_than', cloud_rate) 
        
        if ExtraBandCode == Img.EXTRA_ANGLE:
          L8_toa_ssr_data = Img.SSR_META_DICT['L8_TOA']
          L9_toa_ssr_data = Img.SSR_META_DICT['L9_TOA']
        
          L8_toa_coll = ee.ImageCollection(L8_toa_ssr_data['GEE_NAME']).filterBounds(region).filterDate(start, end).filterMetadata(L8_toa_ssr_data['CLOUD'], 'less_than', cloud_rate) 
          L9_toa_coll = ee.ImageCollection(L9_toa_ssr_data['GEE_NAME']).filterBounds(region).filterDate(start, end).filterMetadata(L9_toa_ssr_data['CLOUD'], 'less_than', cloud_rate) 

          L8_sr_coll = L8_sr_coll.map(lambda img: img.linkCollection(L8_toa_coll, ['SZA', 'SAA', 'VZA', 'VAA']))
          L9_sr_coll = L9_sr_coll.map(lambda img: img.linkCollection(L9_toa_coll, ['SZA', 'SAA', 'VZA', 'VAA']))

        coll = L8_sr_coll.merge(L9_sr_coll)
      else:
        L8_toa_ssr_data = Img.SSR_META_DICT['L8_TOA']
        L9_toa_ssr_data = Img.SSR_META_DICT['L9_TOA']
        
        L8_toa_coll = ee.ImageCollection(L8_toa_ssr_data['GEE_NAME']).filterBounds(region).filterDate(start, end) 
        L9_toa_coll = ee.ImageCollection(L9_toa_ssr_data['GEE_NAME']).filterBounds(region).filterDate(start, end) 

        coll = L8_toa_coll.merge(L9_toa_coll)

  elif ssr_code == Img.HLS_sensor:  # For harmonized Landsat and Sentinel-2
    coll = ee.ImageCollection(CollName).filterBounds(region) \
                                       .filterDate(start, end) \
                                       .filterMetadata(SsrData['CLOUD'], 'less_than', cloud_rate) \
                                       #.filterMetadata(SsrData['SZA'], 'less_than', 70.0) 

  print('\n<getCollection> The name of data catalog = ', CollName)             
  print('<getCollection> The number of images in selected image collection = ', coll.size().getInfo())

  return coll 
  '''