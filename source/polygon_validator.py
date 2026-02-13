"""
Polygon Validation and Logging Module
Filters zero-area polygons and creates processing logs

Usage in Production.py:
    from polygon_validator import filter_valid_polygons, create_processing_log
"""

import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np

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
    
    Args:
        coordinates: GeoJSON polygon coordinates
        
    Returns:
        Area in square meters (approximate, using WGS84)
    """
    try:
        if SHAPELY_AVAILABLE:
            # Create a Shapely polygon
            poly = Polygon(coordinates[0])
            
            # For WGS84 coordinates, we need to convert to a projected CRS
            # For a rough estimate, we can use the area in degrees squared
            # and convert to approximate meters
            area_degrees = poly.area
            
            # Approximate conversion at mid-latitudes (very rough)
            # 1 degree latitude ≈ 111 km, 1 degree longitude varies by latitude
            # This is just for validation purposes
            if area_degrees == 0:
                return 0.0
                
            # Get centroid latitude for better longitude conversion
            centroid_lat = poly.centroid.y
            meters_per_degree_lat = 111000
            meters_per_degree_lon = 111000 * abs(float(centroid_lat) / 90.0)
            
            # Convert to square meters (rough approximation)
            area_m2 = area_degrees * meters_per_degree_lat * meters_per_degree_lon
            
            return area_m2
        else:
            # Fallback: Use shoelace formula for area calculation
            coords = coordinates[0]
            n = len(coords)
            if n < 3:
                return 0.0
            
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += coords[i][0] * coords[j][1]
                area -= coords[j][0] * coords[i][1]
            
            area = abs(area) / 2.0
            
            # Rough conversion to square meters (approximate)
            avg_lat = sum(c[1] for c in coords) / n
            meters_per_degree_lat = 111000
            meters_per_degree_lon = 111000 * abs(avg_lat / 90.0)
            
            area_m2 = area * meters_per_degree_lat * meters_per_degree_lon
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
            'area_m2': area,
            'status': 'SKIPPED' if is_zero else 'VALID',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        log_entries.append(log_entry)
        
        if not is_zero:
            valid_regions[region_id] = region_data
        else:
            print(f"Skipping {region_id}: {reason}")
    
    print(valid_regions)
    # Create DataFrame or list depending on pandas availability
    if PANDAS_AVAILABLE:
        log_df = pd.DataFrame(log_entries)
    else:
        log_df = log_entries  # Return list if pandas not available
    
    return valid_regions, log_df


def create_processing_log(regions: Dict[str, Dict], 
                         end_dates: List[str],
                         output_path: str = 'polygon_processing_log.csv') -> Any:
    """
    Create a comprehensive processing log for all regions and dates.
    
    Args:
        regions: Dictionary of region geometries
        end_dates: List of end dates for processing
        output_path: Path to save the log CSV
        
    Returns:
        DataFrame or list containing the processing log
    """
    log_entries = []
    
    for region_id, region_data in regions.items():
        is_zero, area, reason = is_zero_area_polygon(region_data)
        
        # Create entry for each date
        for end_date in end_dates:
            log_entry = {
                'region_id': region_id,
                'area_m2': round(area, 2),
                'will_process': not is_zero,
                'status': 'SKIPPED' if is_zero else 'QUEUED',
                'skip_reason': reason if is_zero else 'N/A',
                'timestamp': datetime.now().isoformat()
            }
            log_entries.append(log_entry)
    
    # Save to CSV if pandas available
    if PANDAS_AVAILABLE:
        log_df = pd.DataFrame(log_entries)
        log_df.to_csv(output_path, index=False)
        print(f"\nProcessing log saved to: {output_path}")
        
        # Print summary
        total_entries = len(log_df)
        skipped = (log_df['status'] == 'SKIPPED').sum()
        valid = (log_df['status'] == 'QUEUED').sum()
    else:
        # Manual CSV writing if pandas not available
        import csv
        with open(output_path, 'w', newline='') as f:
            if log_entries:
                writer = csv.DictWriter(f, fieldnames=log_entries[0].keys())
                writer.writeheader()
                writer.writerows(log_entries)
        
        log_df = log_entries
        print(f"\nProcessing log saved to: {output_path}")
        
        # Print summary
        total_entries = len(log_entries)
        skipped = sum(1 for e in log_entries if e['status'] == 'SKIPPED')
        valid = sum(1 for e in log_entries if e['status'] == 'QUEUED')
    
    print(f"\nSummary:")
    print(f"   Total region-date combinations: {total_entries}")
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


def update_processing_log(log_df: Any, 
                         region_id: str, 
                         status: str,
                         output_file: str = None,
                         error_message: str = None) -> Any:
    """
    Update the processing log with actual processing results.
    
    Args:
        log_df: Existing processing log (DataFrame or list)
        region_id: Region identifier
        date: Processing date
        status: New status (e.g., 'COMPLETED', 'FAILED')
        output_file: Path to output file if successful
        error_message: Error message if failed
        
    Returns:
        Updated DataFrame or list
    """
    if PANDAS_AVAILABLE and isinstance(log_df, pd.DataFrame):
        mask = (log_df['region_id'] == region_id) & (log_df['date'] == date)
        
        log_df.loc[mask, 'status'] = status
        log_df.loc[mask, 'completed_timestamp'] = datetime.now().isoformat()
        
        if output_file:
            log_df.loc[mask, 'output_file'] = output_file
        
        if error_message:
            log_df.loc[mask, 'error_message'] = error_message
    else:
        # Update list entries
        for entry in log_df:
            if entry['region_id'] == region_id and entry['date'] == date:
                entry['status'] = status
                entry['completed_timestamp'] = datetime.now().isoformat()
                if output_file:
                    entry['output_file'] = output_file
                if error_message:
                    entry['error_message'] = error_message
    
    return log_df


# Example usage and testing run python polygon_validator.py in source file location with terminal
if __name__ == "__main__":
    test_regions = {
        'region0': {
            'type': 'Polygon',
            'coordinates': [[
                [-81.01372554313386, 44.792170087139056],
                [-81.01372554313386, 44.792170087139056],
                [-81.01372554313386, 44.792170087139056],
                [-81.01372554313386, 44.792170087139056]
            ]]
        },
        'region1': {
            'type': 'Polygon',
            'coordinates': [[
                [-80.28730276089648, 44.35311299523343],
                [-80.28730276089648, 44.35311299523343],
                [-80.28730276089648, 44.35311299523343],
                [-80.28730276089648, 44.35311299523343]
            ]]
        },
        'region2': {
            'type': 'Polygon',
            'coordinates': [[
                [-80.0, 44.0],
                [-80.0, 45.0],
                [-81.0, 45.0],
                [-81.0, 44.0],
                [-80.0, 44.0]
            ]]
        }
    }
    
    test_dates = ['2025-08-31']
    
    # Create processing log
    log_df = create_processing_log(test_regions, test_dates, 'test_log.csv')
    
    # Filter valid polygons
    valid_regions, validation_log = filter_valid_polygons(test_regions)
    
    print(f"\nValid regions: {len(valid_regions)}")
    print(f"Invalid regions: {len(test_regions) - len(valid_regions)}")