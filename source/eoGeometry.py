

#==================================================================================================
# define a spatial region around Ottawa
#==================================================================================================
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


tile55_922 = {
    'type': 'Polygon',
    'coordinates': [
       [
         [-77.6221, 47.5314], 
         [-73.8758, 46.7329],
         [-75.0742, 44.2113], 
         [-78.6303, 44.9569],
         [-77.6221, 47.5314]
       ]
    ]
}



#############################################################################################################
# Description: This function returns two lists containing latitudes and longitudes, respectively, from a
#              given geographic region.
#
# Revision history:  2024-May-28  Lixin Sun  Initial creation
# 
#############################################################################################################
def get_region_bbox(inRegion):
  '''Returns two lists containing latitudes and longitudes, respectively, from a given geographic region.
     Args:
        inRegion(dictionary): a given geographic region'''
  
  coords = inRegion['coordinates'][0]
  nCoords = len(coords)
     
  longs = []
  lats  = []
  
  if nCoords > 0:
    for i in range(nCoords):
      longs.append(coords[i][0]) 
      lats.append(coords[i][1])

  return longs, lats




#############################################################################################################
# Description: This function returns a boundary box [min(longs), min(lats), max(longs), max(lats)] of a given 
#              geographic region.
#
# Revision history:  2024-May-28  Lixin Sun  Initial creation
# 
#############################################################################################################
def get_region_bbox(inRegion):
  lons, lats = get_region_bbox(inRegion)
  
  return [min(lons), min(lats), max(lons), max(lats)]




#############################################################################################################
# Description: This function returns central geographic coordinates of a given geographic region.
#
# Revision history:  2024-May-28  Lixin Sun  Initial creation
# 
#############################################################################################################
def get_region_centre(inRegion):
  lons, lats = get_region_bbox(inRegion)
  
  return sum(lons)/len(lons), sum(lats)/len(lats)




#############################################################################################################
# Description: This function divides the bbox associated with a given geographic region (defined by 
#              Lats/Longs) into a number of sub-bboxes
#
# Revision history:  2024-May-27  Lixin Sun  Initial creation
# 
#############################################################################################################
def divide_region(inRegion, nDivides):
  if nDivides <= 0:
    nDivides = 2

  # Obtain the bbox of the given geographic region 
  bbox = get_region_bbox(inRegion)
  left_lon = bbox[0]
  btom_Lat = bbox[1]

  lon_delta = (bbox[2] - left_lon)/nDivides
  lat_delta = (bbox[3] - btom_Lat)/nDivides

  sub_regions = []  
  for i in range(nDivides):    
    for j in range(nDivides):
      sub_region = []
      BL_lon = left_lon+i*lon_delta
      BL_lat = btom_Lat+j*lat_delta

      sub_region.append([BL_lon, BL_lat])
      sub_region.append([BL_lon+lon_delta, BL_lat])
      sub_region.append([BL_lon+lon_delta, BL_lat+lat_delta])
      sub_region.append([BL_lon, BL_lat+lat_delta])
      sub_region.append([BL_lon, BL_lat])
  
      sub_regions.append(sub_region)
  
  return sub_regions





