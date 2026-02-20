"""
prepare_params.py

Pre-production parameter preparation module.
Handles all parameter processing and validation before calling Production.py main().

This module orchestrates:
1. Region loading from KML/SHP files
2. Temporal window generation (4 modes)
3. Polygon validation and filtering

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
from Production import ProdParams
import source.leaf_wrapper as leafWrapper
from source.polygon_validator import (
    filter_valid_polygons,
    create_processing_log,
    is_zero_area_polygon
)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def validate_production_params(ProdParams: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate all production parameters before processing.
    
    Performs comprehensive validation on:
    - Region file paths and formats
    - File variables list structure
    - Index ranges
    - Buffer parameters
    - Temporal parameters
    
    Args:
        ProdParams: Production parameters dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: Boolean indicating if all validations passed
        - error_messages: List of validation error descriptions
    """
    errors = []
    
    # ============ VALIDATE REGIONS (FILE PATH) ============
    regions = ProdParams.get('regions')
    
    if regions is None:
        errors.append("Missing required parameter: 'regions'")
    elif isinstance(regions, str):
        # Check if it's a valid file path with .kml or .shp extension
        regions_path = Path(regions)
        
        if not regions_path.suffix.lower() in ['.kml', '.shp']:
            errors.append(
                f"Invalid region file type: '{regions_path.suffix}'. "
                f"Must be .kml or .shp file"
            )
        
        if not regions_path.exists():
            errors.append(f"Region file not found: '{regions}'")
    elif not isinstance(regions, dict):
        errors.append(
            f"Invalid 'regions' type: {type(regions).__name__}. "
            f"Must be file path (str) or region dictionary"
        )
    
    # ============ VALIDATE FILE_VARIABLES ============
    if 'file_variables' in ProdParams:
        file_vars = ProdParams['file_variables']
        
        if not isinstance(file_vars, list):
            errors.append(
                f"Invalid 'file_variables' type: {type(file_vars).__name__}. "
                f"Must be a list"
            )
        elif len(file_vars) != 3:
            errors.append(
                f"Invalid 'file_variables' length: {len(file_vars)}. "
                f"Must contain exactly 3 elements [id, start_date, end_date]"
            )
        elif not all(isinstance(v, (str, type(None))) for v in file_vars):
            errors.append(
                f"Invalid 'file_variables' content. "
                f"All elements must be strings (column names) or None"
            )
        elif file_vars[0] is None:
            errors.append(
                f"Invalid 'file_variables': First element (id column) cannot be None. "
                f"Date columns (elements 2 and 3) can be None."
            )
    # ============ VALIDATE REGIONS_START_INDEX ============
    start_idx = ProdParams.get('regions_start_index', 0)
    
    if not isinstance(start_idx, int):
        errors.append(
            f"Invalid 'regions_start_index' type: {type(start_idx).__name__}. "
            f"Must be an integer"
        )
    elif start_idx < 0:
        errors.append(
            f"Invalid 'regions_start_index' value: {start_idx}. "
            f"Must be 0 or greater"
        )
    
    # ============ VALIDATE REGIONS_END_INDEX ============
    end_idx = ProdParams.get('regions_end_index')
    
    if end_idx is not None:
        if not isinstance(end_idx, int):
            errors.append(
                f"Invalid 'regions_end_index' type: {type(end_idx).__name__}. "
                f"Must be an integer or None"
            )
        elif end_idx < start_idx:
            errors.append(
                f"Invalid index range: regions_end_index ({end_idx}) must be "
                f"greater than regions_start_index ({start_idx})"
            )
    
    # ============ VALIDATE SPATIAL_BUFFER_M ============
    if 'spatial_buffer_m' in ProdParams:
        spatial_buffer = ProdParams['spatial_buffer_m']
        
        if not isinstance(spatial_buffer, (int, float)):
            errors.append(
                f"Invalid 'spatial_buffer_m' type: {type(spatial_buffer).__name__}. "
                f"Must be a number (int or float)"
            )
    
    # ============ VALIDATE TEMPORAL_BUFFER ============
    if 'temporal_buffer' in ProdParams:
        temporal_buffer = ProdParams['temporal_buffer']
        
        if not isinstance(temporal_buffer, list):
            errors.append(
                f"Invalid 'temporal_buffer' type: {type(temporal_buffer).__name__}. "
                f"Must be a list"
            )
        else:
            # Validate each buffer entry
            for i, buffer_entry in enumerate(temporal_buffer):
                if not isinstance(buffer_entry, (list, tuple)):
                    errors.append(
                        f"Invalid temporal_buffer[{i}]: {type(buffer_entry).__name__}. "
                        f"Each entry must be a list or tuple"
                    )
                    continue
                
                if len(buffer_entry) != 2:
                    errors.append(
                        f"Invalid temporal_buffer[{i}] length: {len(buffer_entry)}. "
                        f"Each entry must have exactly 2 elements"
                    )
                    continue
                
                # Check if it's integer buffer or date buffer
                start_val, end_val = buffer_entry
                
                # Check if both are integers (day offsets)
                if isinstance(start_val, int) and isinstance(end_val, int):
                    continue  # Valid integer buffer
                
                # Check if both are date strings
                elif isinstance(start_val, str) and isinstance(end_val, str):
                    try:
                        start_date = datetime.strptime(start_val, "%Y-%m-%d")
                        end_date = datetime.strptime(end_val, "%Y-%m-%d")
                        
                        if end_date <= start_date:
                            errors.append(
                                f"Invalid temporal_buffer[{i}]: end date '{end_val}' "
                                f"must be after start date '{start_val}'"
                            )
                    except ValueError as e:
                        errors.append(
                            f"Invalid temporal_buffer[{i}] date format: {e}. "
                            f"Dates must be in YYYY-MM-DD format"
                        )
                else:
                    errors.append(
                        f"Invalid temporal_buffer[{i}]: mixed types. "
                        f"Both elements must be integers (day offsets) OR "
                        f"both must be date strings (YYYY-MM-DD)"
                    )
    
    # ============ VALIDATE NUM_YEARS ============
    if 'num_years' in ProdParams:
        num_years = ProdParams['num_years']
        
        if not isinstance(num_years, int):
            errors.append(
                f"Invalid 'num_years' type: {type(num_years).__name__}. "
                f"Must be an integer"
            )
        elif num_years < 1:
            errors.append(
                f"Invalid 'num_years' value: {num_years}. "
                f"Must be 1 or greater"
            )
    
    # Return validation results
    is_valid = len(errors) == 0
    return is_valid, errors


#############################################################################################################
# REGION HANDLING
#############################################################################################################
def handle_regions_from_file(ProdParams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle KML or SHP file-based regions input.
    
    Converts file-based region definitions to LEAF-compatible region dictionary.
    Applies temporal buffer to region-specific start and end dates if specified.  # ← UPDATED DOCSTRING
    
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

            # Unpack tuple of regions and dates
            regions_dict, region_start_dates, region_end_dates = leafWrapper.regions_from_kml(
                regions,
                start=ProdParams.get("regions_start_index", 0),
                end=ProdParams.get("regions_end_index", None),
                spatial_buffer_m=ProdParams.get("spatial_buffer_m", None),
                file_variables=ProdParams.get("file_variables", None)
            )
            
            # ==================== HANDLE EMPTY REGIONS CASE ====================
            if not regions_dict:
                print(f"\n{'='*80}")
                print("  WARNING: No valid regions loaded from file")
                print('='*80)
                print("This typically occurs when negative buffer operations collapse all geometries.")
                print("The processing pipeline will continue with an empty region set.")
                print(f"{'='*80}\n")
                
                # Set empty region parameters and continue
                ProdParams["regions"] = {}
                ProdParams["region_start_dates"] = {}
                ProdParams["region_end_dates"] = {}
                
                return ProdParams
            
            # ==================== TEMPORAL BUFFER PROCESSING ====================

            # STEP 1: Handle single date case FIRST (auto-generate missing dates)
            all_regions = set(regions_dict.keys())

            for region_name in all_regions:
                has_start = region_name in region_start_dates and region_start_dates[region_name]
                has_end = region_name in region_end_dates and region_end_dates[region_name]
                
                if has_start and not has_end:
                    # Only start date exists - copy to end
                    region_end_dates[region_name] = region_start_dates[region_name].copy()
                    print(f"<handle_regions_from_file> Region {region_name}: Auto-generated end dates from start dates")
                elif has_end and not has_start:
                    # Only end date exists - copy to start
                    region_start_dates[region_name] = region_end_dates[region_name].copy()
                    print(f"<handle_regions_from_file> Region {region_name}: Auto-generated start dates from end dates")

            # STEP 2: Apply temporal buffer (now that all regions have both start and end dates)
            if 'temporal_buffer' in ProdParams:
                buffer_list = ProdParams['temporal_buffer']
                
                # Determine if we're in override mode or offset mode
                # Override mode: buffer_list contains pairs of date strings like [["2024-04-15", "2024-07-15"], ["2024-08-01", "2024-09-01"]]
                # Offset mode: buffer_list contains pairs of day offsets like [[-5, 10], [-10, 15], [0, 30]]
                
                if buffer_list and len(buffer_list) > 0:
                    # Check if first entry is a date string pair or numeric offset pair
                    first_entry = buffer_list[0] if isinstance(buffer_list[0], list) else buffer_list
                    is_override_mode = isinstance(first_entry[0], str)
                    
                    # Normalize to list of pairs format
                    if not isinstance(buffer_list[0], list):
                        # Single pair like [-5, 10] or ["2024-04-15", "2024-07-15"]
                        buffer_list = [buffer_list]
                    
                    if is_override_mode:
                        # OVERRIDE MODE: Replace all region dates with specified date pairs
                        print(f"<handle_regions_from_file> Override mode: Setting all regions to {len(buffer_list)} date window(s)")
                        
                        new_start_dates = {}
                        new_end_dates = {}
                        
                        for region_name in regions_dict.keys():
                            new_start_dates[region_name] = [pair[0] for pair in buffer_list]
                            new_end_dates[region_name] = [pair[1] for pair in buffer_list]
                            print(f"<handle_regions_from_file> Region {region_name}: {len(buffer_list)} windows {buffer_list}")
                        
                        region_start_dates = new_start_dates
                        region_end_dates = new_end_dates
                    
                    else:
                        # OFFSET MODE: Apply each buffer pair to existing dates
                        print(f"<handle_regions_from_file> Offset mode: Applying {len(buffer_list)} buffer(s) to multiply date windows")
                        
                        new_start_dates = {}
                        new_end_dates = {}
                        
                        for region_name in regions_dict.keys():
                            expanded_starts = []
                            expanded_ends = []
                            
                            # Get original dates for this region
                            orig_starts = region_start_dates.get(region_name, [])
                            orig_ends = region_end_dates.get(region_name, [])
                            
                            # If no dates exist, skip this region
                            if not orig_starts and not orig_ends:
                                continue
                            
                            # For each original date pair
                            for i in range(max(len(orig_starts), len(orig_ends))):
                                orig_start = orig_starts[i] if i < len(orig_starts) else orig_ends[i]
                                orig_end = orig_ends[i] if i < len(orig_ends) else orig_starts[i]
                                
                                # Apply each buffer to this date pair
                                for buffer_start, buffer_end in buffer_list:
                                    buffered_start = apply_temporal_buffer_to_date(orig_start, buffer_start)
                                    buffered_end = apply_temporal_buffer_to_date(orig_end, buffer_end)
                                    expanded_starts.append(buffered_start)
                                    expanded_ends.append(buffered_end)
                            
                            new_start_dates[region_name] = expanded_starts
                            new_end_dates[region_name] = expanded_ends
                            print(f"<handle_regions_from_file> Region {region_name}: Expanded from {max(len(orig_starts), len(orig_ends))} to {len(expanded_starts)} windows")
                        
                        region_start_dates = new_start_dates
                        region_end_dates = new_end_dates

            # STEP 3: Expand region dates across multiple years if num_years is specified 
            # (MOVED AFTER buffering so buffered dates get expanded across years)
            if 'num_years' in ProdParams and ProdParams['num_years'] > 1:
                num_years = ProdParams['num_years']
                print(f"<handle_regions_from_file> Expanding region dates across {num_years} years...")
                
                expanded_start_dates = {}
                expanded_end_dates = {}
                
                for region_name in all_regions:
                    orig_starts = region_start_dates.get(region_name, [])
                    orig_ends = region_end_dates.get(region_name, [])
                    
                    if not orig_starts or not orig_ends:
                        continue
                    
                    multi_year_starts = []
                    multi_year_ends = []
                    
                    # For each original date pair (now potentially buffered)
                    for orig_start, orig_end in zip(orig_starts, orig_ends):
                        # Parse the dates
                        start_dt = datetime.strptime(orig_start, '%Y-%m-%d')
                        end_dt = datetime.strptime(orig_end, '%Y-%m-%d')
                        
                        # Generate dates for each year
                        for year_offset in range(num_years):
                            # Add years, handling leap year edge case
                            try:
                                new_start = start_dt.replace(year=start_dt.year + year_offset)
                                new_end = end_dt.replace(year=end_dt.year + year_offset)
                            except ValueError:
                                # Handle Feb 29 in non-leap years → move to Feb 28
                                if start_dt.month == 2 and start_dt.day == 29:
                                    new_start = start_dt.replace(year=start_dt.year + year_offset, day=28)
                                else:
                                    new_start = start_dt.replace(year=start_dt.year + year_offset)
                                
                                if end_dt.month == 2 and end_dt.day == 29:
                                    new_end = end_dt.replace(year=end_dt.year + year_offset, day=28)
                                else:
                                    new_end = end_dt.replace(year=end_dt.year + year_offset)
                            
                            multi_year_starts.append(new_start.strftime('%Y-%m-%d'))
                            multi_year_ends.append(new_end.strftime('%Y-%m-%d'))
                    
                    expanded_start_dates[region_name] = multi_year_starts
                    expanded_end_dates[region_name] = multi_year_ends
                    print(f"<handle_regions_from_file> Region {region_name}: Expanded from {len(orig_starts)} to {len(multi_year_starts)} date windows ({num_years} years)")
                
                region_start_dates = expanded_start_dates
                region_end_dates = expanded_end_dates

            # ==================== END TEMPORAL BUFFER PROCESSING ====================
            ProdParams["regions"] = regions_dict
            ProdParams["region_start_dates"] = region_start_dates
            ProdParams["region_end_dates"] = region_end_dates
            
            print(f"<handle_regions_from_file> Loaded {len(ProdParams['regions'])} regions from file")
            
            return ProdParams

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
# TEMPORAL WINDOW GENERATION HELPER
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
                    ProdParams['start_dates'].append(start)
                    ProdParams['end_dates'].append(end)
            
            # Apply temporal buffer if specified (only for numeric offsets, not date overrides)
            if 'temporal_buffer' in ProdParams:
                buffer_list = ProdParams['temporal_buffer']
                # Normalize to list of pairs
                if not isinstance(buffer_list[0], list):
                    buffer_list = [buffer_list]
                
                # Check if it's numeric offset mode (not date override mode)
                first_entry = buffer_list[0]
                if isinstance(first_entry[0], int):
                    # Only apply buffer if we have a single numeric offset pair
                    if len(buffer_list) == 1:
                        buffer_start, buffer_end = buffer_list[0]
                        ProdParams['start_dates'] = [apply_temporal_buffer_to_date(d, buffer_start) for d in ProdParams['start_dates']]
                        ProdParams['end_dates'] = [apply_temporal_buffer_to_date(d, buffer_end) for d in ProdParams['end_dates']]
                        print(f"<form_time_windows> Applied temporal buffer: [{buffer_start}, {buffer_end}] days")
                    else:
                        print(f"<form_time_windows> Warning: Multiple buffers ignored for global time windows (use region dates instead)")
        
        else:
            # MODE 1: Single year, multiple months
            print(f"<form_time_windows> Mode 1: Single year months ({ProdParams['year']})")
            
            nMonths = len(ProdParams['months'])
            year = ProdParams['year']
            ProdParams['start_dates'] = []
            ProdParams['end_dates'] = []
            
            for month in ProdParams['months']:
                start, end = month_range(year, month)
                ProdParams['start_dates'].append(start)
                ProdParams['end_dates'].append(end)
            
            # Apply temporal buffer if specified (only for numeric offsets, not date overrides)
            if 'temporal_buffer' in ProdParams:
                buffer_list = ProdParams['temporal_buffer']
                # Normalize to list of pairs
                if not isinstance(buffer_list[0], list):
                    buffer_list = [buffer_list]
                
                # Check if it's numeric offset mode (not date override mode)
                first_entry = buffer_list[0]
                if isinstance(first_entry[0], int):
                    # Only apply buffer if we have a single numeric offset pair
                    if len(buffer_list) == 1:
                        buffer_start, buffer_end = buffer_list[0]
                        ProdParams['start_dates'] = [apply_temporal_buffer_to_date(d, buffer_start) for d in ProdParams['start_dates']]
                        ProdParams['end_dates'] = [apply_temporal_buffer_to_date(d, buffer_end) for d in ProdParams['end_dates']]
                        print(f"<form_time_windows> Applied temporal buffer: [{buffer_start}, {buffer_end}] days")
                    else:
                        print(f"<form_time_windows> Warning: Multiple buffers ignored for global time windows (use region dates instead)")
            
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
            buffer_list = ProdParams['temporal_buffer']
            # Normalize to list of pairs
            if not isinstance(buffer_list[0], list):
                buffer_list = [buffer_list]
            
            # Check if it's numeric offset mode (not date override mode)
            first_entry = buffer_list[0]
            if isinstance(first_entry[0], int):
                # Only apply buffer if we have a single numeric offset pair
                if len(buffer_list) == 1:
                    buffer_start, buffer_end = buffer_list[0]
                    ProdParams['start_dates'] = [apply_temporal_buffer_to_date(d, buffer_start) for d in ProdParams['start_dates']]
                    ProdParams['end_dates'] = [apply_temporal_buffer_to_date(d, buffer_end) for d in ProdParams['end_dates']]
                    print(f"<form_time_windows> Applied temporal buffer: [{buffer_start}, {buffer_end}] days")
                else:
                    print(f"<form_time_windows> Warning: Multiple buffers ignored for global time windows (use region dates instead)")
        
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

def apply_global_dates_to_regions(ProdParams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply global start_dates and end_dates to regions that have empty date values.
    
    This ensures that regions loaded from files without date attributes can still
    use the global temporal windows defined in ProdParams.
    
    Args:
        ProdParams: Production parameters dictionary
        
    Returns:
        Updated ProdParams with global dates applied to empty region dates
    """
    # Check if we have global dates and regions
    if 'start_dates' not in ProdParams or 'end_dates' not in ProdParams:
        return ProdParams
    
    if 'regions' not in ProdParams or not isinstance(ProdParams['regions'], dict):
        return ProdParams
    
    global_starts = ProdParams['start_dates']
    global_ends = ProdParams['end_dates']
    
    # Initialize region date dictionaries if they don't exist
    if 'region_start_dates' not in ProdParams:
        ProdParams['region_start_dates'] = {}
    if 'region_end_dates' not in ProdParams:
        ProdParams['region_end_dates'] = {}
    
    region_start_dates = ProdParams['region_start_dates']
    region_end_dates = ProdParams['region_end_dates']
    
    # Apply global dates to regions with empty dates
    regions_updated = 0
    for region_name in ProdParams['regions'].keys():
        has_start = region_name in region_start_dates and region_start_dates[region_name]
        has_end = region_name in region_end_dates and region_end_dates[region_name]
        
        # If region has no dates at all, apply global dates
        if not has_start and not has_end:
            region_start_dates[region_name] = global_starts.copy()
            region_end_dates[region_name] = global_ends.copy()
            regions_updated += 1
            print(f"<apply_global_dates_to_regions> Region '{region_name}': Applied {len(global_starts)} global time windows")
    
    if regions_updated > 0:
        print(f"<apply_global_dates_to_regions> Updated {regions_updated} region(s) with global time windows")
    
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
    
    # Also filter region-specific dates to match valid regions
    if 'region_start_dates' in ProdParams:
        filtered_start_dates = {k: v for k, v in ProdParams['region_start_dates'].items() if k in valid_regions}
        ProdParams['region_start_dates'] = filtered_start_dates
    
    if 'region_end_dates' in ProdParams:
        filtered_end_dates = {k: v for k, v in ProdParams['region_end_dates'].items() if k in valid_regions}
        ProdParams['region_end_dates'] = filtered_end_dates
    
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
# MAIN FUNCTION
#############################################################################################################

def prepare_production_params(
    ProdParams: Dict[str, Any],
    CompParams: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Orchestrate all parameter preparation steps before production.
    
    This is the main entry point that runs all preprocessing:
    0. Validate input parameters
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
    print("PARAMETER PREPARATION")
    print("="*80 + "\n")
    
    # Step 0: Validate parameters
    print("Step 1/4: Validating input parameters...")
    is_valid, errors = validate_production_params(ProdParams)
    
    if not is_valid:
        print("\n" + "="*80)
        print("PARAMETER VALIDATION FAILED")
        print("="*80)
        print(f"\nFound {len(errors)} validation error(s):\n")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\n" + "="*80)
        print("Please fix the above errors and try again.")
        print("="*80 + "\n")
        return None
    
    print("✓ All parameters valid")
    
    # Step 1: Handle file-based regions (KML/SHP)
    print("\nStep 2/4: Loading regions from files...")
    ProdParams = handle_regions_from_file(ProdParams)
    
    ## Step 2: Generate temporal windows (only if temporal parameters exist)
    # Check if any temporal parameters are present
    has_year_months = 'year' in ProdParams and 'months' in ProdParams
    has_custom_dates = has_custom_window(ProdParams)
    has_standardized = 'standardized' in ProdParams
    
    if has_year_months or has_custom_dates or has_standardized:
        print("\nStep 3/4: Generating temporal windows...")
        ProdParams = form_time_windows(ProdParams)
        
        # Step 2.5: Apply global dates to regions with empty dates
        if 'regions' in ProdParams and isinstance(ProdParams['regions'], dict):
            ProdParams = apply_global_dates_to_regions(ProdParams)
    else:
        print("\nStep 3/4: Skipping temporal window generation (no year/months or start_dates/end_dates found)")
        print("           Region-specific dates from file will be used if available.")
    
    # Step 3: Validate and filter polygons
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
    print(f"Regions: {len(ProdParams.get('regions', {}))}")
    print(f"Start Time windows: {ProdParams.get('region_start_dates', [])}")
    print(f"End Time windows: {ProdParams.get('region_end_dates', [])}")
    print("="*80 + "\n")
    
    return {
        'ProdParams': ProdParams,
        'CompParams': CompParams,
        'processing_log': processing_log
    }