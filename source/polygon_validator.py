"""
Polygon Validation Module
Pre-processing module that filters zero-area polygons and creates validation logs.

Usage in prepare_params.py:
    from source.polygon_validator import filter_valid_polygons, create_processing_log
"""

from typing import Dict, List, Tuple, Any
import math

# Optional imports - shapely is preferred but not required
try:
    from shapely.geometry import Polygon, shape
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def calculate_polygon_area(coordinates: List[List[List[float]]]) -> float:
    """
    Calculate the area of a polygon in square meters.
    Uses Shapely if available, otherwise uses shoelace formula.
    
    For WGS84 coordinates, approximates by converting to meters using
    the centroid latitude and assuming Mercator-like projection.
    
    Args:
        coordinates: GeoJSON polygon coordinates [[lon, lat], [lon, lat], ...]
        
    Returns:
        Area in square meters (approximate)
    """
    try:
        if SHAPELY_AVAILABLE:
            # Create a Shapely polygon from WGS84 coordinates
            poly = Polygon(coordinates[0])
            
            if poly.area == 0:
                return 0.0
            
            # Get centroid latitude for accurate longitude scaling
            centroid_lat = poly.centroid.y
            centroid_lat_rad = math.radians(centroid_lat)
            
            # Convert degrees² to square meters using proper projection
            # 1 degree latitude ≈ 111,320 meters (constant)
            # 1 degree longitude ≈ 111,320 * cos(latitude) meters (varies by latitude)
            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * math.cos(centroid_lat_rad)
            
            # Area in square meters
            # Note: area_degrees is in degree² units
            area_m2 = poly.area * meters_per_degree_lat * meters_per_degree_lon
            
            return area_m2
        else:
            # Fallback: Use shoelace formula for area in degrees²
            coords = coordinates[0]
            n = len(coords)
            if n < 3:
                return 0.0
            
            # Calculate area using shoelace formula
            area_degrees = 0.0
            for i in range(n):
                j = (i + 1) % n
                area_degrees += coords[i][0] * coords[j][1]
                area_degrees -= coords[j][0] * coords[i][1]
            
            area_degrees = abs(area_degrees) / 2.0
            
            if area_degrees == 0:
                return 0.0
            
            # Get centroid latitude for conversion
            avg_lat = sum(c[1] for c in coords) / n
            avg_lat_rad = math.radians(avg_lat)
            
            # Convert to square meters
            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * math.cos(avg_lat_rad)
            
            area_m2 = area_degrees * meters_per_degree_lat * meters_per_degree_lon
            return area_m2
        
    except Exception as e:
        print(f"Error calculating area: {e}")
        return 0.0


def is_zero_area_polygon(region_data: Dict) -> Tuple[bool, float, str]:
    """
    Check if a polygon has zero or near-zero area.
    
    Args:
        region_data: Dictionary containing polygon geometry
        
    Returns:
        Tuple of (is_zero_area, area_m2, reason)
    """
    try:
        coordinates = region_data['coordinates']
        
        # Check if all coordinates are identical (degenerate polygon)
        first_coord = coordinates[0][0]
        all_identical = all(
            coord == first_coord 
            for coord in coordinates[0]
        )
        
        if all_identical:
            return True, 0.0, "All coordinates identical (point, not polygon)"
        
        # Check if we have fewer than 3 unique points (minimum for polygon)
        unique_coords = list(set(tuple(coord) for coord in coordinates[0]))
        if len(unique_coords) < 3:
            return True, 0.0, f"Only {len(unique_coords)} unique coordinate(s)"
        
        # Calculate actual area
        area = calculate_polygon_area(coordinates)
        
        # Define threshold for "zero area" (e.g., less than 1 square meter)
        area_threshold = 1.0  # square meters
        
        if area < area_threshold:
            return True, area, f"Area ({area:.2f} m²) below threshold ({area_threshold} m²)"
        
        return False, area, "Valid polygon"
        
    except Exception as e:
        return True, 0.0, f"Error validating polygon: {str(e)}"


def filter_valid_polygons(regions: Dict[str, Dict]) -> Tuple[Dict[str, Dict], Any]:
    """
    Filter out zero-area polygons and create a validation log.
    
    Args:
        regions: Dictionary of region geometries
        
    Returns:
        Tuple of (valid_regions_dict, validation_log_df or list)
    """
    valid_regions = {}
    log_entries = []
    
    for region_id, region_data in regions.items():
        is_zero, area, reason = is_zero_area_polygon(region_data)
        
        log_entry = {
            'region_id': region_id,
            'area_m2': round(area, 2),
            'status': 'SKIPPED' if is_zero else 'VALID',
            'reason': reason
        }
        
        log_entries.append(log_entry)
        
        if not is_zero:
            valid_regions[region_id] = region_data
        else:
            print(f"Skipping {region_id}: {reason}")
    
    # Create DataFrame or list depending on pandas availability
    if PANDAS_AVAILABLE:
        log_df = pd.DataFrame(log_entries)
    else:
        log_df = log_entries  # Return list if pandas not available
    
    return valid_regions, log_df


def create_processing_log(regions: Dict[str, Dict], 
                         labels: List[str] = None,
                         output_path: str = 'polygon_processing_log.csv') -> Any:
    """
    Create a pre-processing validation log for all regions.
    
    Args:
        regions: Dictionary of region geometries
        labels: Optional list of labels/names (unused, for compatibility)
        output_path: Path to save the log CSV
        
    Returns:
        DataFrame or list containing the validation log
    """
    log_entries = []
    
    for region_id, region_data in regions.items():
        is_zero, area, reason = is_zero_area_polygon(region_data)
        
        log_entry = {
            'region_id': region_id,
            'area_m2': round(area, 2),
            'will_process': not is_zero,
            'status': 'SKIPPED' if is_zero else 'VALID',
            'skip_reason': reason if is_zero else 'N/A'
        }
        log_entries.append(log_entry)
    
    # Save to CSV if pandas available
    if PANDAS_AVAILABLE:
        log_df = pd.DataFrame(log_entries)
        log_df.to_csv(output_path, index=False)
        print(f"\nPolygon validation log saved to: {output_path}")
        
        # Print summary
        total_entries = len(log_df)
        skipped = (log_df['status'] == 'SKIPPED').sum()
        valid = (log_df['status'] == 'VALID').sum()
    else:
        # Manual CSV writing if pandas not available
        import csv
        with open(output_path, 'w', newline='') as f:
            if log_entries:
                writer = csv.DictWriter(f, fieldnames=log_entries[0].keys())
                writer.writeheader()
                writer.writerows(log_entries)
        
        log_df = log_entries
        print(f"\nPolygon validation log saved to: {output_path}")
        
        # Print summary
        total_entries = len(log_entries)
        skipped = sum(1 for e in log_entries if e['status'] == 'SKIPPED')
        valid = sum(1 for e in log_entries if e['status'] == 'VALID')
    
    print(f"\nValidation Summary:")
    print(f"   Total regions: {total_entries}")
    print(f"   Valid for processing: {valid}")
    print(f"   Skipped (zero area): {skipped}")
    
    if skipped > 0:
        print(f"\nZero-area regions detected:")
        zero_area_regions = set()
        for entry in log_entries if not PANDAS_AVAILABLE else log_df[log_df['status'] == 'SKIPPED']['region_id'].unique():
            if not PANDAS_AVAILABLE:
                if entry['status'] == 'SKIPPED':
                    zero_area_regions.add(entry['region_id'])
            else:
                zero_area_regions.add(entry)
        
        for region_id in zero_area_regions:
            print(f"   - {region_id}")
    
    return log_df