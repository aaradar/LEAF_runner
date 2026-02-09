"""
prepare_params.py

Pre-production parameter preparation module.
Handles all parameter processing and validation before calling Production.py main().

This module orchestrates:
1. Region loading from KML/SHP files
2. Temporal window generation (4 modes)
3. Date symmetry and ordering
4. Polygon validation and filtering

Usage:
    from prepare_params import prepare_production_params
    
    validated_params = prepare_production_params(ProdParams, CompParams)
    if validated_params:
        main(validated_params['ProdParams'], validated_params['CompParams'])
"""

import os
import copy
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Import required modules
import leaf_wrapper as leafWrapper
from polygon_validator import (
    filter_valid_polygons,
    create_processing_log,
    is_zero_area_polygon
)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


#############################################################################################################
# REGION HANDLING
#############################################################################################################

def handle_regions_from_file(ProdParams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle KML or SHP file-based regions input.
    
    Converts file-based region definitions to LEAF-compatible region dictionary.
    
    Args:
        ProdParams: Production parameters dictionary
        
    Returns:
        Updated ProdParams dictionary with regions loaded from file
    """
    try:
        regions = ProdParams.get("regions")

        if isinstance(regions, str) and regions.lower().endswith((".kml", ".shp")):
            # Check if file exists before attempting to load
            if not os.path.exists(regions):
                raise FileNotFoundError(
                    f"Region file not found: {regions}\n"
                    f"Current working directory: {os.getcwd()}\n"
                    f"Please provide a valid file path."
                )
            
            print(f"<handle_regions_from_file> Detected file-based regions input: {regions}")
            print("<handle_regions_from_file> Loading regions...")

            ProdParams["regions"] = leafWrapper.regions_from_kml(
                regions,
                start=ProdParams.get("regions_start_index", 0),
                end=ProdParams.get("regions_end_index", None),
                spatial_buffer_m=ProdParams.get("spatial_buffer_m", None)
            )
            
            print(f"<handle_regions_from_file> Loaded {len(ProdParams['regions'])} regions from file")

    except FileNotFoundError as e:
        # Re-raise with better error message
        print(f"\n{'='*80}")
        print("ERROR: Region file not found")
        print('='*80)
        print(str(e))
        print('='*80 + "\n")
        raise
        
    except (AttributeError, TypeError) as e:
        # regions is None or not a string — skip silently
        print(f"<handle_regions_from_file> No file-based regions to process: {e}")
        pass

    return ProdParams


#############################################################################################################
# TEMPORAL WINDOW GENERATION (4 MODES)
#############################################################################################################

def apply_temporal_buffer_to_date(date_str: Any, buffer_days: int) -> Any:
    """
    Apply a temporal buffer (in days) to a date string or datetime object.
    
    Args:
        date_str: Either a string in 'YYYY-MM-DD' format or a datetime object
        buffer_days: Number of days to add (can be negative)
        
    Returns:
        Same type as input (string or datetime object) with buffer applied
    """
    if isinstance(date_str, str):
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        buffered_dt = dt + timedelta(days=buffer_days)
        return buffered_dt.strftime('%Y-%m-%d')
    else:
        return date_str + timedelta(days=buffer_days)


def month_range(year: int, month: int) -> Tuple[str, str]:
    """
    Get the first and last day of a given month.
    
    Args:
        year: Year (e.g., 2023)
        month: Month number (1-12)
        
    Returns:
        Tuple of (start_date, end_date) as 'YYYY-MM-DD' strings
    """
    # First day is always the 1st
    start_date = f"{year}-{month:02d}-01"
    
    # Last day depends on the month
    if month in [1, 3, 5, 7, 8, 10, 12]:
        last_day = 31
    elif month in [4, 6, 9, 11]:
        last_day = 30
    else:  # February
        # Check for leap year
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            last_day = 29
        else:
            last_day = 28
    
    end_date = f"{year}-{month:02d}-{last_day:02d}"
    
    return start_date, end_date


def has_custom_window(inParams: Dict[str, Any]) -> bool:
    """
    Check if custom time windows are defined.
    
    Args:
        inParams: Input parameters dictionary
        
    Returns:
        True if custom windows exist, False otherwise
    """
    has_start = 'start_dates' in inParams and len(inParams.get('start_dates', [])) > 0
    has_end = 'end_dates' in inParams and len(inParams.get('end_dates', [])) > 0
    has_single_start = 'start_date' in inParams
    has_single_end = 'end_date' in inParams
    
    return has_start or has_end or has_single_start or has_single_end


def form_time_windows(ProdParams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create start and end dates based on temporal mode.
    
    Supports 4 temporal modes:
    1. Single month(s): months + year
    2. Custom date range: start_dates + end_dates
    3. Single date: start_date OR end_date (auto-generates the other)
    4. Multi-year months: months + year + num_years
    
    Args:
        ProdParams: Production parameters dictionary
        
    Returns:
        Updated ProdParams with start_dates and end_dates populated
    """
    print("<form_time_windows> Generating temporal windows...")
    
    if not has_custom_window(ProdParams):
        # MODE 1 & 4: Month-based windows
        ProdParams['monthly'] = True

        # Check if we're in "repeated months for X years" mode
        if 'num_years' in ProdParams and ProdParams['num_years'] > 1:
            # MODE 4: Multi-year monthly mosaics
            print(f"<form_time_windows> Mode 4: Multi-year months ({ProdParams['num_years']} years)")
            
            nMonths = len(ProdParams['months'])
            num_years = ProdParams['num_years']
            start_year = ProdParams['year']
            ProdParams['start_dates'] = []
            ProdParams['end_dates'] = []
            
            # Loop through each year
            for year_offset in range(num_years):
                current_year = start_year + year_offset
                # Loop through each month for this year
                for month in ProdParams['months']:
                    start, end = month_range(current_year, month)
                    
                    # Apply temporal buffer if specified
                    if 'temporal_buffer' in ProdParams:
                        buffer_start = ProdParams['temporal_buffer'][0]
                        buffer_end = ProdParams['temporal_buffer'][1]
                        start = apply_temporal_buffer_to_date(start, buffer_start)
                        end = apply_temporal_buffer_to_date(end, buffer_end)
                    
                    ProdParams['start_dates'].append(start)
                    ProdParams['end_dates'].append(end)
            
            print(f"<form_time_windows> Generated {len(ProdParams['start_dates'])} time windows")
        
        else:
            # MODE 1: Single year, multiple months
            print(f"<form_time_windows> Mode 1: Single year months ({ProdParams['year']})")
            
            nMonths = len(ProdParams['months'])
            year = ProdParams['year']
            ProdParams['start_dates'] = []
            ProdParams['end_dates'] = []
            
            for month in ProdParams['months']:
                start, end = month_range(year, month)
                
                # Apply temporal buffer if specified
                if 'temporal_buffer' in ProdParams:
                    buffer_start = ProdParams['temporal_buffer'][0]
                    buffer_end = ProdParams['temporal_buffer'][1]
                    start = apply_temporal_buffer_to_date(start, buffer_start)
                    end = apply_temporal_buffer_to_date(end, buffer_end)
                
                ProdParams['start_dates'].append(start)
                ProdParams['end_dates'].append(end)
            
            print(f"<form_time_windows> Generated {len(ProdParams['start_dates'])} time windows")
    
    elif 'standardized' not in ProdParams:
        # MODE 2 & 3: Custom date windows
        ProdParams['monthly'] = False
        
        # Handle single date inputs (Mode 3)
        if 'start_date' in ProdParams and 'start_dates' not in ProdParams:
            ProdParams['start_dates'] = [ProdParams['start_date']]
        if 'end_date' in ProdParams and 'end_dates' not in ProdParams:
            ProdParams['end_dates'] = [ProdParams['end_date']]
        
        # Handle cases where only start_dates or end_dates exist
        has_start = 'start_dates' in ProdParams and len(ProdParams['start_dates']) > 0
        has_end = 'end_dates' in ProdParams and len(ProdParams['end_dates']) > 0
        
        if has_start and not has_end:
            # MODE 3: Only start_dates exists - copy to end_dates
            print("<form_time_windows> Mode 3: Single start date (auto-generating end_dates)")
            ProdParams['end_dates'] = ProdParams['start_dates'].copy()
        elif has_end and not has_start:
            # MODE 3: Only end_dates exists - copy to start_dates
            print("<form_time_windows> Mode 3: Single end date (auto-generating start_dates)")
            ProdParams['start_dates'] = ProdParams['end_dates'].copy()
        else:
            # MODE 2: Both exist
            print("<form_time_windows> Mode 2: Custom date range")
        
        # Apply temporal buffer to custom windows if needed
        if 'temporal_buffer' in ProdParams and 'start_dates' in ProdParams and 'end_dates' in ProdParams:
            buffer_start = ProdParams['temporal_buffer'][0]
            buffer_end = ProdParams['temporal_buffer'][1]
            ProdParams['start_dates'] = [apply_temporal_buffer_to_date(d, buffer_start) for d in ProdParams['start_dates']]
            ProdParams['end_dates'] = [apply_temporal_buffer_to_date(d, buffer_end) for d in ProdParams['end_dates']]
            print(f"<form_time_windows> Applied temporal buffer: [{buffer_start}, {buffer_end}] days")
        
        # Ensure start_date is always before end_date (swap if needed)
        if 'start_dates' in ProdParams and 'end_dates' in ProdParams:
            for i in range(len(ProdParams['start_dates'])):
                start_dt = datetime.strptime(ProdParams['start_dates'][i], '%Y-%m-%d') if isinstance(ProdParams['start_dates'][i], str) else ProdParams['start_dates'][i]
                end_dt = datetime.strptime(ProdParams['end_dates'][i], '%Y-%m-%d') if isinstance(ProdParams['end_dates'][i], str) else ProdParams['end_dates'][i]
                
                if start_dt > end_dt:
                    # Swap the dates
                    print(f"<form_time_windows> Swapping dates for window {i}: {ProdParams['start_dates'][i]} <-> {ProdParams['end_dates'][i]}")
                    ProdParams['start_dates'][i], ProdParams['end_dates'][i] = ProdParams['end_dates'][i], ProdParams['start_dates'][i]
        
        print(f"<form_time_windows> Generated {len(ProdParams.get('start_dates', []))} time windows")

    ProdParams['current_time'] = 0
    return ProdParams


#############################################################################################################
# DATE SYMMETRY
#############################################################################################################

def ensure_date_symmetry(ProdParams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure start_dates and end_dates are symmetric.
    
    If only one is provided, the other is set to match it.
    
    Args:
        ProdParams: Production parameters dictionary
        
    Returns:
        Updated ProdParams dictionary with symmetric dates
    """
    start_dates = ProdParams.get("start_dates")
    end_dates = ProdParams.get("end_dates")

    if start_dates is not None and end_dates is None:
        ProdParams["end_dates"] = copy.deepcopy(start_dates)
        print("<ensure_date_symmetry> end_dates set to match start_dates")

    elif end_dates is not None and start_dates is None:
        ProdParams["start_dates"] = copy.deepcopy(end_dates)
        print("<ensure_date_symmetry> start_dates set to match end_dates")

    return ProdParams


#############################################################################################################
# POLYGON VALIDATION
#############################################################################################################

def validate_and_filter_polygons(
    ProdParams: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    """
    Validate and filter zero-area polygons before processing.
    
    Creates a processing log, filters out invalid polygons, and updates
    the ProdParams with only valid regions.
    
    Args:
        ProdParams: Production parameters dictionary
        
    Returns:
        Tuple of (updated ProdParams, processing_log DataFrame)
        Returns (None, None) if all polygons are invalid
    """
    regions = ProdParams.get('regions', {})
    
    if not regions or not isinstance(regions, dict):
        print("<validate_and_filter_polygons> No regions to validate")
        return ProdParams, None
    
    print('\n' + '='*80)
    print('POLYGON VALIDATION AND FILTERING')
    print('='*80)
    
    # Get end dates for logging
    end_dates_list = _get_end_dates_for_logging(ProdParams)
    
    # Create output folder if needed
    out_folder = ProdParams.get('out_folder')
    if out_folder:
        os.makedirs(out_folder, exist_ok=True)
    
    # Create initial processing log
    log_path = os.path.join(out_folder or '.', 'polygon_processing_log.csv')
    processing_log = create_processing_log(
        regions,
        end_dates_list if end_dates_list else ['processing'],
        output_path=log_path
    )
    
    # Filter valid polygons
    valid_regions, validation_log = filter_valid_polygons(regions)
    
    # Check if we have any valid regions
    if len(valid_regions) == 0:
        print("\n" + "="*80)
        print("ERROR: All polygons have zero area!")
        print("="*80)
        print("\nNo valid polygons to process. Please check your input data.")
        print("All regions have identical coordinates (degenerate polygons).")
        print(f"\nProcessing log saved to: {log_path}")
        return None, processing_log
    
    # Update ProdParams with only valid regions
    ProdParams['regions'] = valid_regions
    
    print(f"\nProceeding with {len(valid_regions)} valid region(s) out of {len(regions)} total")
    print("="*80 + "\n")
    
    return ProdParams, processing_log


def _get_end_dates_for_logging(ProdParams: Dict[str, Any]) -> List[str]:
    """
    Helper function to extract or construct end dates for logging.
    
    Args:
        ProdParams: Production parameters dictionary
        
    Returns:
        List of end date strings
    """
    end_dates_list = ProdParams.get('end_dates', [])
    
    if not end_dates_list and 'year' in ProdParams and 'months' in ProdParams:
        # Construct end dates from year/months if needed
        year = ProdParams['year']
        months = ProdParams['months']
        end_dates_list = [f"{year}-{month:02d}-28" for month in months]
    
    return end_dates_list


#############################################################################################################
# MAIN ORCHESTRATION FUNCTION
#############################################################################################################

def prepare_production_params(
    ProdParams: Dict[str, Any],
    CompParams: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Orchestrate all parameter preparation steps before production.
    
    This is the main entry point that runs all preprocessing:
    1. Load regions from KML/SHP files (if applicable)
    2. Generate temporal windows (4 modes supported)
    3. Ensure date symmetry
    4. Validate and filter polygons
    
    Args:
        ProdParams: Production parameters dictionary
        CompParams: Computation parameters dictionary
        
    Returns:
        Dictionary containing validated parameters or None if validation fails:
        {
            'ProdParams': validated_prod_params,
            'CompParams': comp_params,
            'processing_log': log_dataframe
        }
    """
    print("\n" + "="*80)
    print("PARAMETER PREPARATION PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Handle file-based regions (KML/SHP)
    print("Step 1/4: Loading regions from files...")
    ProdParams = handle_regions_from_file(ProdParams)
    
    # Step 2: Generate temporal windows
    print("\nStep 2/4: Generating temporal windows...")
    ProdParams = form_time_windows(ProdParams)
    
    # Step 3: Ensure date symmetry
    print("\nStep 3/4: Ensuring date symmetry...")
    ProdParams = ensure_date_symmetry(ProdParams)
    
    # Step 4: Validate and filter polygons
    print("\nStep 4/4: Validating polygons...")
    ProdParams, processing_log = validate_and_filter_polygons(ProdParams)
    
    if ProdParams is None:
        print("\n" + "="*80)
        print("PARAMETER PREPARATION FAILED")
        print("="*80)
        print("No valid polygons to process. Aborting production.")
        return None
    
    print("\n" + "="*80)
    print("PARAMETER PREPARATION COMPLETE")
    print("="*80)
    print(f"✓ Regions: {len(ProdParams.get('regions', {}))}")
    print(f"✓ Time windows: {len(ProdParams.get('start_dates', []))}")
    print(f"✓ Total tasks: {len(ProdParams.get('regions', {})) * len(ProdParams.get('start_dates', []))}")
    print("="*80 + "\n")
    
    return {
        'ProdParams': ProdParams,
        'CompParams': CompParams,
        'processing_log': processing_log
    }


#############################################################################################################
# EXAMPLE USAGE
#############################################################################################################

if __name__ == "__main__":
    # Example: Test the preprocessing pipeline
    import sys
    
    # Check if a KML file path was provided as command line argument
    if len(sys.argv) > 1:
        kml_file = sys.argv[1]
    else:
        # Default test path - adjust this to your actual KML file location
        base_dir = Path(__file__).parent
        kml_file = base_dir / "Sample Points" / "AfforestationSItesFixed.kml"
    
    # Convert to string and check if file exists
    kml_file = str(kml_file)
    if not os.path.exists(kml_file):
        print(f"ERROR: KML file not found: {kml_file}")
        print("\nUsage:")
        print(f"  python {Path(__file__).name} <path_to_kml_file>")
        print("\nExample:")
        print(f"  python {Path(__file__).name} ./Sample\\ Points/AfforestationSItesFixed.kml")
        sys.exit(1)
    
    print(f"Using KML file: {kml_file}")
    
    test_params = {
        'sensor': 'HLS_SR',
        'year': 2023,
        'months': [6, 7],
        'regions': kml_file,
        'regions_start_index': 0,
        'regions_end_index': 5,
        'resolution': 30,
        'projection': 'EPSG:3979',
        'out_folder': './test_output'
    }
    
    test_comp = {
        'number_workers': 10,
        'debug': True
    }
    
    result = prepare_production_params(test_params, test_comp)
    
    if result:
        print("\n✓ Preprocessing successful!")
        print(f"Ready to call: main(ProdParams, CompParams)")
        
        # Print summary
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        print(f"Regions loaded: {len(result['ProdParams']['regions'])}")
        print(f"Time windows: {len(result['ProdParams']['start_dates'])}")
        print(f"Start dates: {result['ProdParams']['start_dates']}")
        print(f"End dates: {result['ProdParams']['end_dates']}")
        
        # Show first few regions as examples
        region_names = list(result['ProdParams']['regions'].keys())
        print(f"\nRegion names: {region_names[:3]}...")
        
        # Show first region coordinates (first 3 points)
        if result['ProdParams']['regions']:
            first_region_name = region_names[0]
            coords = result['ProdParams']['regions'][first_region_name]['coordinates'][0]
            print(f"\nExample region '{first_region_name}' (first 3 coordinates):")
            for i, coord in enumerate(coords[:3]):
                print(f"  Point {i+1}: {coord}")
        
        print("="*80)
    else:
        print("\n✗ Preprocessing failed!")
        sys.exit(1)