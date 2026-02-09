"""
Test script for negative buffer functionality in leaf_wrapper.py
Tests how negative buffers handle geometry collapse scenarios
"""

from pathlib import Path
import json
from source.leaf_wrapper import regions_from_kml

# Get absolute paths
base_dir = Path(__file__).parent
kml_path = base_dir / "Sample Points" / "AfforestationSItesFixed.kml"
shp_path = base_dir / "Sample Points" / "FieldPoints32_2018.shp"

# Test 1: Small negative buffer (should reduce polygons slightly)
print("="*80)
print("TEST 1: KML with SMALL NEGATIVE BUFFER (-50m)")
print("="*80)

try:
    kml_regions_small = regions_from_kml(str(kml_path), start=0, end=2, spatial_buffer_m=-50)
    print(f"[OK] Loaded {len(kml_regions_small)} KML regions with -50m buffer")
    print(f"First region preview:")
    first_key = list(kml_regions_small.keys())[0]
    print(f"  Region: {first_key}")
    print(f"  Coordinates count: {len(kml_regions_small[first_key]['coordinates'][0])} points")
    print()
except Exception as e:
    print(f"[ERROR] {e}")
    print()


# Test 2: Large negative buffer (should collapse some/all polygons to zero area)
print("="*80)
print("TEST 2: KML with LARGE NEGATIVE BUFFER (-500m)")
print("="*80)
print("This should collapse small polygons to zero-area points")
print()

try:
    kml_regions_large = regions_from_kml(str(kml_path), start=0, end=4, spatial_buffer_m=-500)
    print(f"[OK] Loaded {len(kml_regions_large)} KML regions with -500m buffer")
    
    # Check each region for collapse
    for region_name, region_data in kml_regions_large.items():
        coords = region_data['coordinates'][0]
        
        # Check if all points are the same (collapsed geometry)
        unique_coords = set(tuple(pt) for pt in coords)
        if len(unique_coords) == 1:
            print(f"  {region_name}: COLLAPSED to zero area at {coords[0]}")
        else:
            print(f"  {region_name}: Still has area ({len(coords)} points)")
    print()
except Exception as e:
    print(f"[ERROR] {e}")
    print()


# Test 3: Shapefile with negative buffer on point geometries
print("="*80)
print("TEST 3: SHAPEFILE POINTS with POSITIVE then NEGATIVE BUFFER")
print("="*80)
print("First apply +200m buffer to points, then -150m to shrink them")
print()

try:
    # First create buffered points
    shp_regions_pos = regions_from_kml(str(shp_path), start=0, end=2, spatial_buffer_m=200)
    print(f"[OK] Step 1: Created {len(shp_regions_pos)} point buffers at +200m")
    
    for region_name, region_data in shp_regions_pos.items():
        coords = region_data['coordinates'][0]
        print(f"  {region_name}: {len(coords)} points")
    print()
    
except Exception as e:
    print(f"[ERROR] {e}")
    print()


# Test 4: Extreme negative buffer (should definitely collapse everything)
print("="*80)
print("TEST 4: KML with EXTREME NEGATIVE BUFFER (-2000m)")
print("="*80)
print("This should collapse ALL polygons to zero-area")
print()

try:
    kml_regions_extreme = regions_from_kml(str(kml_path), start=0, end=5, spatial_buffer_m=-2000)
    print(f"[OK] Loaded {len(kml_regions_extreme)} KML regions with -2000m buffer")
    
    # Count collapsed vs non-collapsed
    collapsed_count = 0
    for region_name, region_data in kml_regions_extreme.items():
        coords = region_data['coordinates'][0]
        unique_coords = set(tuple(pt) for pt in coords)
        if len(unique_coords) == 1:
            collapsed_count += 1
    
    print(f"  Collapsed: {collapsed_count}/{len(kml_regions_extreme)} regions")
    print(f"  Still have area: {len(kml_regions_extreme) - collapsed_count}/{len(kml_regions_extreme)} regions")
    print()
    
    # Show example of collapsed geometry
    if collapsed_count > 0:
        print("Example collapsed geometry:")
        for region_name, region_data in kml_regions_extreme.items():
            coords = region_data['coordinates'][0]
            unique_coords = set(tuple(pt) for pt in coords)
            if len(unique_coords) == 1:
                print(f"  {region_name}:")
                print(f"    All points at: {coords[0]}")
                break
    print()
    
except Exception as e:
    print(f"[ERROR] {e}")
    print()


# Test 5: Export full JSON for one collapsed example
print("="*80)
print("TEST 5: FULL JSON OUTPUT - One Normal vs One Collapsed")
print("="*80)

try:
    # Normal polygon (no buffer)
    normal = regions_from_kml(str(kml_path), start=0, end=0, spatial_buffer_m=None)
    print("NORMAL POLYGON (no buffer):")
    print(json.dumps(normal, indent=2))
    print()
    
    # Collapsed polygon (large negative buffer)
    collapsed = regions_from_kml(str(kml_path), start=0, end=0, spatial_buffer_m=-2000)
    print("COLLAPSED POLYGON (-2000m buffer):")
    print(json.dumps(collapsed, indent=2))
    print()
    
except Exception as e:
    print(f"[ERROR] {e}")


print("="*80)
print("TEST COMPLETE")
print("="*80)