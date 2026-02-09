from source.leaf_wrapper import LeafWrapper

def regions_from_kml(kml_file, start=0, end=9, prefix="region", spatial_buffer_m=None):
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

    Returns:
    --------
    tuple: (regions_dict, region_start_dates, region_end_dates)
        - regions_dict: Dictionary mapping region names to GeoJSON Polygon objects
        - region_start_dates: Dictionary mapping region names to [start_date]
        - region_end_dates: Dictionary mapping region names to [end_date]
        
    Example:
        # File has 10 regions with IDs 20, 39, 45, 47, 52, 53, 57, 61, ...
        regions, starts, ends = regions_from_kml('file.kml', start=1, end=5)
        # Returns: 
        #   regions = {'region39': {...}, 'region45': {...}, 'region47': {...}, 'region52': {...}, 'region53': {...}}
        #   starts = {'region39': ['2009-07-01'], 'region45': ['2009-07-01'], ...}
        #   ends = {'region39': ['2010-07-01'], 'region45': ['2010-07-01'], ...}
    """
    if start < 0 or (end is not None and end < start):
        raise ValueError("Invalid start or end values. 'start' must be >= 0 and 'end' must be >= 'start'.")

    # Load the KML file with optional spatial buffer
    wrapper = LeafWrapper(kml_file, spatial_buffer_m=spatial_buffer_m).load()
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

    # Build output dictionaries
    out = {}
    region_start_dates = {}
    region_end_dates = {}
    
    for key in selected_keys:
        region_data = regions_dict[key]
        file_id = region_data.get("r_id", key)
        coords = region_data.get("coordinates", [])
        
        if not coords:
            continue
        
        region_name = f"{prefix}{file_id}"
        
        # Add region geometry
        out[region_name] = {
            "type": "Polygon",
            "coordinates": coords,
        }
        
        # Extract and add start date
        start_date = region_data.get("start_date")
        if start_date is not None and str(start_date).strip() and str(start_date).lower() != 'nan':
            region_start_dates[region_name] = [start_date]
        
        # Extract and add end date
        end_date = region_data.get("end_date")
        if end_date is not None and str(end_date).strip() and str(end_date).lower() != 'nan':
            region_end_dates[region_name] = [end_date]
    
    return out, region_start_dates, region_end_dates


if __name__ == "__main__":
    import json
    
    print("="*80)
    print("Test regions_from_kml:")
    print("="*80)
    
    # Test with first 3 regions
    regions, start_dates, end_dates = regions_from_kml(
        "C:\\Users\\aradar\\LEAF Files\\LEAF_runner\\Sample Points\\AfforestationSItesFixed.kml", 
        start=0, 
        end=20
    )
    
    print(f"\nExtracted {len(regions)} regions")
    print(f"Regions with start dates: {len(start_dates)}")
    print(f"Regions with end dates: {len(end_dates)}")
    
    print("\n" + "="*80)
    print("REGIONS DICTIONARY:")
    print("="*80)
    print("\nregions = {")
    for i, (key, value) in enumerate(sorted(regions.items())):
        coords = value['coordinates']
        coord_count = len(coords[0]) if coords and coords[0] else 0
        
        # Show abbreviated coordinates
        if coords and coords[0] and len(coords[0]) > 0:
            first_coord = coords[0][0]
            last_coord = coords[0][-1]
            coords_preview = f"[[{first_coord}, ... ({coord_count} points), {last_coord}]]"
        else:
            coords_preview = str(coords)
        
        print(f"    '{key}': {{")
        print(f"        'type': '{value['type']}',")
        print(f"        'coordinates': {coords_preview}")
        print(f"    }},")
    print("}")
    
    print("\n" + "="*80)
    print("START DATES DICTIONARY:")
    print("="*80)
    print("\nregion_start_dates = {")
    for key, value in sorted(start_dates.items()):
        print(f"    '{key}': {value},")
    print("}")
    
    print("\n" + "="*80)
    print("END DATES DICTIONARY:")
    print("="*80)
    print("\nregion_end_dates = {")
    for key, value in sorted(end_dates.items()):
        print(f"    '{key}': {value},")
    print("}")
    
    print("\n" + "="*80)
    print("FULL OUTPUT OF FIRST REGION (Complete Coordinates):")
    print("="*80)
    if regions:
        first_key = sorted(regions.keys())[0]
        print(f"\nregions['{first_key}'] = {{")
        print(f"    'type': '{regions[first_key]['type']}',")
        print(f"    'coordinates': [")
        coords = regions[first_key]['coordinates']
        if coords and coords[0]:
            print(f"        [")
            for i, point in enumerate(coords[0]):
                comma = "," if i < len(coords[0]) - 1 else ""
                print(f"            {point}{comma}")
            print(f"        ]")
        print(f"    ]")
        print(f"}}")
        
        if first_key in start_dates:
            print(f"\nregion_start_dates['{first_key}'] = {start_dates[first_key]}")
        
        if first_key in end_dates:
            print(f"\nregion_end_dates['{first_key}'] = {end_dates[first_key]}")
    
    print("\n" + "="*80)
    print("USAGE IN LEAF CODE:")
    print("="*80)
    print("""
# Import the function
from regions_from_kml import regions_from_kml

# Load regions with dates
regions, start_dates, end_dates = regions_from_kml(
    kml_file="path/to/file.kml",
    start=0,
    end=49,
    prefix="region"
)

# Use in LEAF parameters
ProdParams['regions'] = regions
ProdParams['region_start_dates'] = start_dates
ProdParams['region_end_dates'] = end_dates
    """)
    
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)