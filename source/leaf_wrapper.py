import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Union, Optional
from shapely.geometry import mapping
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union, transform
import re

#############################################################################################################
# Description: This function reads a KML or Shapefile containing polygon/point features and converts them
#              into a LEAF-compatible region dictionary. For shapefiles with Point geometries, a spatial
#              buffer can be applied to convert points to polygons. The output dictionary maps user-defined
#              region names to GeoJSON-like Polygon objects, which can be directly passed to
#              ProdParams['regions'] for mosaic generation.
#
# Revision history: 2025-January-19 Alexander Radar Initial creation
#############################################################################################################

def regions_from_kml(kml_file, start=14, end=14, prefix="region", spatial_buffer_m=None, file_variables=None):
    """
    Load a KML or Shapefile and return a dict of polygon regions by file ID.
    Selects regions at positions [start:end+1] from the sorted file.
    Output keys use the actual file IDs (TARGET_FID, point ID, etc.) for identification.

    Parameters:
    -----------
    kml_file : str or Path
        Path to KML or Shapefile
    start : int
        Starting position (0-based, inclusive). Example: start=0 selects the 1st region
    end : int
        Ending position (0-based, inclusive). Example: end=5 selects up to the 6th region
    prefix : str
        Prefix for region names in output
    spatial_buffer_m : float or None
        Buffer size in meters to apply to geometries
    file_variables : list or None
        [id_column, start_date_column, end_date_column] from KML/SHP attributes

    Returns:
    dict
        Regions keyed as "{prefix}{file_id}". Example: {'region20': {...}, 'region39': {...}, 'region45': {...}}
        # Returns: tuple (regions_dict, region_start_dates, region_end_dates)
    Example:
        # File has 10 regions with IDs 20, 39, 45, 47, 52, 53, 57, 61, ...
        regions = regions_from_kml('file.kml', start=1, end=5)
        # Returns: {'region39': {...}, 'region45': {...}, 'region47': {...}, 'region52': {...}, 'region53': {...}}
    """
    if start < 0 or (end is not None and end < start):
        raise ValueError("Invalid start or end values. 'start' must be >= 0 and 'end' must be >= 'start'.")

    wrapper = LeafWrapper(kml_file, spatial_buffer_m=spatial_buffer_m, file_variables=file_variables).load()
    regions_dict = wrapper.to_region_dict()

    # Get sorted keys and select by position
    sorted_keys = sorted(regions_dict.keys())

    # Handle end=None → go to end of file
    if end is None:
        selected_keys = sorted_keys[start:]
    else:
        selected_keys = sorted_keys[start : end + 1]

    if not selected_keys:
        raise ValueError(f"No regions found in range [{start}:{end+1}]. File has {len(sorted_keys)} regions.")

    out = {}
    region_start_dates = {}  # NEW
    region_end_dates = {}    # NEW
    
    for key in selected_keys:
        region_data = regions_dict[key]
        file_id = region_data.get("r_id", key)
        coords = region_data.get("coordinates", [])
        if not coords:
            continue
        
        region_name = f"{prefix}{file_id}"
        
        out[region_name] = {
            "type": "Polygon",
            "coordinates": coords,
        }
        
        # NEW: Extract dates
        start_date = region_data.get("start_date")
        if start_date:
            region_start_dates[region_name] = [start_date]
        else:
            region_start_dates[region_name] = []  # or [] if you prefer empty list for missing dates
        
        end_date = region_data.get("end_date")
        if end_date:
            region_end_dates[region_name] = [end_date]
        else:
            region_end_dates[region_name] = []  # or [] if you prefer empty list for missing dates
    
    return out, region_start_dates, region_end_dates  # Returns tuple of (regions_dict, start_dates_dict, end_dates_dict)


#############################################################################################################
# Class: LeafWrapper
#
# Description:
# ------------
# Wrapper class to load KML or Shapefile polygon/point data into a GeoDataFrame,
# optionally apply a spatial buffer to point or polygon geometries, and convert
# the data into a dictionary format compatible with LEAF mosaic generation.
#
# Features:
#   - Load KML (.kml) or Shapefile (.shp) files
#   - Drop Z coordinates (altitude) to ensure 2D polygons
#   - Apply metric buffer to points or polygons
#   - Convert MultiPolygon geometries to single polygons or bounding boxes
#   - Generate a dictionary suitable for ProdParams['regions']
#
#############################################################################################################

class LeafWrapper:
    def __init__(self, polygon_file, spatial_buffer_m=None, file_variables=['ID', 'start_date', 'end_date']):
        self.polygon_file = Path(polygon_file)
        self.gdf = None
        self.spatial_buffer_m = spatial_buffer_m
        self.file_variables = file_variables

    def load(self):
        """Load the polygon file into a GeoDataFrame"""
        if not self.polygon_file.exists():
            raise FileNotFoundError(f"Polygon file not found: {self.polygon_file}")

        ext = self.polygon_file.suffix.lower()
        if ext == ".kml":
            self.gdf = gpd.read_file(self.polygon_file, driver="KML")
        else:
            self.gdf = gpd.read_file(self.polygon_file)

        if self.gdf.empty:
            raise ValueError("Loaded polygon file contains no geometries.")

        # Drop Z coordinates if present
        self.gdf["geometry"] = self.gdf.geometry.apply(drop_z)

        # Apply spatial buffer if specified
        if self.spatial_buffer_m is not None:
            self._apply_buffer()

        return self

    def _apply_buffer(self):
        """Apply spatial buffer to geometries, converting to appropriate CRS if needed."""
        if self.gdf is None:
            raise ValueError("No data loaded")
        
        original_crs = self.gdf.crs
        
        if self.gdf.crs is None:
            print("Warning: No CRS defined, assuming EPSG:4326")
            self.gdf = self.gdf.set_crs("EPSG:4326")
        
        # Reproject to EPSG:3979 for buffering
        gdf_projected = self.gdf.to_crs("EPSG:3979")
        
        # **FIX: Ensure geometries are valid before buffering**
        gdf_projected['geometry'] = gdf_projected.geometry.apply(
            lambda geom: geom.buffer(0) if not geom.is_valid else geom
        )
        
        def safe_buffer(geom):
            """Apply buffer and handle collapsed geometries"""
            # **FIX: Ensure input geometry is valid**
            if not geom.is_valid:
                geom = geom.buffer(0)
            
            # **FIX: Use resolution parameter for smoother results**
            buffered = geom.buffer(self.spatial_buffer_m, resolution=16)
            
            # If negative buffer collapses geometry, warn but continue
            if buffered.is_empty or buffered.area == 0:
                print(f"Warning: Buffer of {self.spatial_buffer_m}m collapsed geometry. "
                    f"Original area: {geom.area:.2f} m²")
            
            # **FIX: Ensure output is valid**
            if not buffered.is_valid:
                buffered = buffered.buffer(0)
                
            return buffered
        
        try:
            gdf_projected['geometry'] = gdf_projected.geometry.apply(safe_buffer)
        except ValueError as e:
            print(f"ERROR: {e}")
            raise
        
        # Convert back to original CRS
        target_crs = original_crs if original_crs is not None else "EPSG:4326"
        self.gdf = gdf_projected.to_crs(target_crs)
        
        print(f"Applied {self.spatial_buffer_m}m buffer to geometries")

    def to_region_dict(self, use_target_fid: bool = True) -> Dict[int, Dict]:
        """
        Convert loaded geometries into a LEAF-compatible region dictionary.

        Actions performed:
        -----------------
        - Converts all geometries to EPSG:4326
        - Handles Polygon and MultiPolygon geometries
        - Optionally merges MultiPolygons into a single bounding-box polygon
        - Generates a dictionary keyed by TARGET_FID or other ID columns

        Parameters:
        -----------
        use_target_fid : bool
            If True, use "TARGET_FID" column as the dictionary key (if available)

        Returns:
        --------
        regions : dict
            Dictionary mapping region IDs to GeoJSON-like Polygon data:
            Example: {'region20': {"r_id": 20, "coordinates": [[...]]}, ...}

        Raises:
        -------
        ValueError : If GeoDataFrame is not loaded
        """
        if self.gdf is None:
            raise ValueError("No polygon file loaded. Call `.load()` first.")

        # Convert to geographic coordinates (EPSG:4326) for output
        gdf_geo = self.gdf.to_crs("EPSG:4326")
        regions = {}

        # Extract column names from file_variables
        id_col, start_col, end_col = self.file_variables

        for idx, row in gdf_geo.iterrows():
            # Extract region ID - keeps value as-is (string or int)
            if id_col in gdf_geo.columns:
                id_value = row[id_col]
                if id_value is not None:
                    # Convert to int if it's a float (like 20.0 -> 20), otherwise keep as-is
                    if isinstance(id_value, float) and id_value.is_integer():
                        key = int(id_value)
                    else:
                        key = id_value
                else:
                    key = idx
            else:
                key = idx if isinstance(idx, int) else 0
                
            geom = row.geometry
            if geom.is_empty:
                continue

            coords = []
            if geom.geom_type == "Polygon":
                ring = [list(pt) for pt in geom.exterior.coords]
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                coords.append([ring])  # wrap polygon consistently

            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    ring = [list(pt) for pt in poly.exterior.coords]
                    if ring[0] != ring[-1]:
                        ring.append(ring[0])
                    coords.append([ring])  # wrap polygon

                # If there are multiple polygon entries, compute one bounding-box polygon (axis-aligned).
                if len(coords) > 1:
                    shapely_polys = []
                    for poly in coords:
                        # each `poly` is [[ring]] (you only keep the exterior ring)
                        ring = poly[0]
                        p = ShapelyPolygon(ring)
                        if not p.is_valid:
                            p = p.buffer(0)
                        if not p.is_empty:
                            shapely_polys.append(p)

                    if shapely_polys:
                        merged = unary_union(shapely_polys)
                        minx, miny, maxx, maxy = merged.bounds
                        # Build bbox ring (closed)
                        bbox_ring = [
                            [minx, miny],
                            [maxx, miny],
                            [maxx, maxy],
                            [minx, maxy],
                            [minx, miny],
                        ]
                        # Replace coords with a single polygon: [[bbox_ring]]
                        coords = [[bbox_ring]]
            
            # Handle LineString and MultiLineString (CONVERT TO POLYGON VIA BUFFER)
            elif geom.geom_type in ["LineString", "MultiLineString"]:
                # Buffer must be applied - LineStrings can't be regions without buffering
                if self.spatial_buffer_m is None:
                    print(f"Warning: Skipping LineString {key} - requires spatial_buffer_m parameter")
                    continue
                
                # Buffer was already applied in _apply_buffer(), so this geometry should now be a Polygon
                # If it's still a LineString here, something went wrong
                print(f"Warning: LineString {key} was not converted to Polygon by buffer operation")
                continue
            
            # Handle GeometryCollection (extract and convert sub-geometries)
            elif geom.geom_type == "GeometryCollection":
                for sub_geom in geom.geoms:
                    if sub_geom.geom_type == "Polygon":
                        ring = [list(pt) for pt in sub_geom.exterior.coords]
                        if ring[0] != ring[-1]:
                            ring.append(ring[0])
                        coords.append([ring])
            
            # Skip if no valid polygon coordinates were generated
            if not coords:
                print(f"Warning: Skipping geometry {key} - type {geom.geom_type} could not be converted to polygon")
                continue
            
            # Extract date fields (only if columns exist)
            # Extract date fields (only if columns exist)
            start_date = None
            end_date = None
            if start_col and start_col in gdf_geo.columns:
                start_date = row.get(start_col)
                # Convert Timestamp to string if needed
                if hasattr(start_date, 'strftime'):
                    start_date = start_date.strftime('%Y-%m-%d')
                
            if end_col and end_col in gdf_geo.columns:
                end_date = row.get(end_col)
                # Convert Timestamp to string if needed
                if hasattr(end_date, 'strftime'):
                    end_date = end_date.strftime('%Y-%m-%d')
            
            # Package region data
            regions[key] = {
                "r_id": key,
                "coordinates": coords[0],
                "start_date": start_date,  # NEW
                "end_date": end_date        # NEW
            }

        return regions


#############################################################################################################
# Function: drop_z
#
# Description:
# ------------
# Removes the Z coordinate (altitude) from a Shapely geometry if present.
# Converts 3D coordinates (x, y, z) to 2D (x, y), which is required for LEAF regions.
# If the geometry has no Z coordinate, it is returned unchanged.
#############################################################################################################

def drop_z(geom):
    """Remove the Z coordinate (altitude) from a Shapely geometry"""
    if geom.has_z:
        return transform(lambda x, y, z=None: (x, y), geom)
    return geom