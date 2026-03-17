"""
s2_tile_processor.py

Sentinel-2 tile processing module.
Handles mode='tiles' conversion by finding all S2 MGRS tiles that intersect
selected regions and returning their tile footprints instead of the regions themselves.

Supported tile grid formats:
    - .parquet  (sentinel-2-grid.parquet from maawoo/sentinel-2-grid-geoparquet)
    - .kml      (ESA S2A_OPER_GIP_TILPAR KML — parsed via xml.etree, no fiona required)
    - any other format readable by geopandas (gpd.read_file fallback)

Usage:
    from source.s2_tile_processor import resolve_s2_tiles

    # After loading regions with leaf_wrapper in 'regions' mode
    if mode == 'tiles':
        tiles_dict, start_dates, end_dates = resolve_s2_tiles(
            regions_dict=regions_dict,
            selected_keys=selected_keys,
            s2_grid_path=s2_grid_path
        )
"""

import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple
from shapely.geometry import shape
from shapely.ops import unary_union

# Default path to the Sentinel-2 tile grid file — expected alongside this script.
# Download sentinel-2-grid.parquet from: https://github.com/maawoo/sentinel-2-grid-geoparquet
_DEFAULT_S2_GRIDS = {
    "parquet": Path(__file__).parent / "sentinel-2-grid.parquet",
    "kml":     Path(__file__).parent / "S2A_OPER_GIP_TILPAR_MPC.kml",
}


def resolve_s2_tiles(
    regions_dict: Dict,
    selected_keys: List,
    s2_grid_path=_DEFAULT_S2_GRIDS["parquet"],
    keep_first_match_only: bool = False,
) -> Tuple[Dict, Dict, Dict]:
    """
    Find all Sentinel-2 MGRS tiles that intersect the selected regions and
    return their footprints clipped to the convex hull of the source regions
    that touch each tile.

    Parameters:
    -----------
    regions_dict : dict
        Full output from LeafWrapper.to_region_dict()
    selected_keys : list
        Subset of keys from regions_dict to process
    s2_grid_path : str, Path, or None
        Path to the S2 tile grid file (.parquet, .kml, or any geopandas-readable
        format). Defaults to sentinel-2-grid.parquet next to this script.
    keep_first_match_only : bool, default False
        If True, for each source region keep only the first (primary) intersecting
        tile. This avoids returning multiple overlapping MGRS tiles for a single
        region. If False (default), all intersecting tiles are returned.

    Returns:
    --------
    (out_dict, start_dates_dict, end_dates_dict)
        out_dict keys: "tile_{MGRS_NAME}"  e.g. "tile_18TXR"
        Each tile geometry is clipped to the convex hull of all source regions
        that intersect it, so no excess area is included.
        Dates are aggregated from all source regions that touch each tile.
        Coordinate rings contain only (lon, lat) — no Z values.
    """
    # ── 1. Build GeoDataFrame from the selected regions ────────────────────
    geom_rows = []
    for key in selected_keys:
        rd = regions_dict[key]
        coords = rd.get("coordinates", [])
        if not coords:
            continue
        try:
            geom = shape({"type": "Polygon", "coordinates": coords})
            if not geom.is_valid:
                geom = geom.buffer(0)
            geom_rows.append({
                "region_key":  key,
                "geometry":    geom,
                "start_dates": rd.get("start_dates", []),
                "end_dates":   rd.get("end_dates",   []),
            })
        except Exception as e:
            print(f"  Warning: Could not parse geometry for region {key}: {e}")

    if not geom_rows:
        print("  WARNING: No valid geometries to resolve S2 tiles for.")
        return {}, {}, {}

    regions_gdf = gpd.GeoDataFrame(geom_rows, crs="EPSG:4326")

    # ── 2. Load Sentinel-2 tile grid ───────────────────────────────────────
    if s2_grid_path is not None:
        grid_path = Path(__file__).parent / s2_grid_path
    else:
        # Try defaults in order: parquet first (faster), then KML
        grid_path = next(
            (p for p in _DEFAULT_S2_GRIDS.values() if p.exists()),
            None
        )
        if grid_path is None:
            raise FileNotFoundError(
                f"No S2 tile grid file found. Looked for:\n"
                + "\n".join(f"  {p}" for p in _DEFAULT_S2_GRIDS.values())
                + "\nDownload one and place it next to this script, or pass s2_grid_path=<path>."
            )

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Sentinel-2 tile grid not found at: {grid_path}\n"
            f"Supported formats:\n"
            f"  .parquet — download from https://github.com/maawoo/sentinel-2-grid-geoparquet\n"
            f"  .kml     — ESA S2A_OPER_GIP_TILPAR KML (no fiona KML driver required)\n"
            f"Place the file next to this script or pass s2_grid_path=<path>."
        )

    print(f"  Loading S2 tile grid from: {grid_path}")

    suffix = grid_path.suffix.lower()
    if suffix == ".parquet":
        s2_grid = _load_parquet(grid_path)
    elif suffix == ".kml":
        s2_grid = _load_kml(grid_path)
    else:
        # Fallback: any format geopandas supports (GeoJSON, Shapefile, GPKG, …)
        s2_grid = gpd.read_file(grid_path)
        print(f"    Loaded via geopandas ({len(s2_grid)} tiles)")

    if s2_grid.crs is None:
        s2_grid = s2_grid.set_crs("EPSG:4326")
    else:
        s2_grid = s2_grid.to_crs("EPSG:4326")

    # ── 3. Pre-filter grid to bounding box of all regions (fast pre-filter) ─
    minx, miny, maxx, maxy = regions_gdf.total_bounds
    margin = 0.5  # degrees — small padding to catch edge-touching tiles
    s2_local = s2_grid.cx[minx - margin : maxx + margin, miny - margin : maxy + margin].copy()
    print(f"  Pre-filtered to {len(s2_local)} candidate S2 tiles in bounding box")

    if s2_local.empty:
        print("  WARNING: No Sentinel-2 tiles found in the bounding box of the selected regions.")
        return {}, {}, {}

    # ── 4. Spatial join: which tiles intersect any selected region? ─────────
    joined = gpd.sjoin(
        s2_local,
        regions_gdf[["geometry", "start_dates", "end_dates"]],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        print("  WARNING: No Sentinel-2 tiles intersect the selected regions.")
        return {}, {}, {}

    # ── 4b. Optional: keep only first match per source region ───────────────
    if keep_first_match_only:
        print("  Filtering to first match only (one tile per source region)...")
        joined = joined.groupby("index_right").first()
        joined.reset_index(inplace=True)

    # ── 5. Auto-detect the MGRS tile name column ───────────────────────────
    tile_col = _find_tile_name_column(s2_local)
    print(f"  Tile name column: '{tile_col}'")

    # ── 6. Build output: one entry per unique intersecting tile ────────────
    # Each tile is clipped to the convex hull of all source regions touching it,
    # so only the relevant area within the tile is retained.
    out             = {}
    start_dates_out = {}
    end_dates_out   = {}

    # Set of tile names confirmed to intersect at least one selected region
    matched_tile_names = set(joined[tile_col].astype(str).unique())

    for _, tile_row in s2_local.iterrows():
        tile_name = str(tile_row[tile_col])
        if tile_name not in matched_tile_names:
            continue

        tile_geom = tile_row.geometry
        if tile_geom is None or tile_geom.is_empty:
            continue

        # ── Clip tile to convex hull of all source regions touching this tile ──
        touching = joined[joined[tile_col].astype(str) == tile_name]
        touching_region_indices = (
            touching["index_right"].values
            if "index_right" in touching.columns
            else touching.index.values
        )
        touching_region_geoms = regions_gdf.loc[
            regions_gdf.index.isin(touching_region_indices), "geometry"
        ].tolist()

        if touching_region_geoms:
            convex_hull = unary_union(touching_region_geoms).convex_hull
            tile_geom = tile_geom.intersection(convex_hull)

            # Alternative (use actual region shapes instead of convex hull):
            # clip_geom = unary_union(touching_region_geoms).buffer(1e-6)
            # tile_geom = tile_geom.intersection(clip_geom)

            if tile_geom is None or tile_geom.is_empty:
                print(f"  Skipping tile {tile_name}: empty after clipping to convex hull")
                continue

        # ── Extract exterior ring — 2D only (lon, lat), no Z ──────────────
        if tile_geom.geom_type == "Polygon":
            ring = [[pt[0], pt[1]] for pt in tile_geom.exterior.coords]
        elif tile_geom.geom_type == "MultiPolygon":
            # Use the largest sub-polygon
            largest = max(tile_geom.geoms, key=lambda p: p.area)
            ring = [[pt[0], pt[1]] for pt in largest.exterior.coords]
        else:
            print(f"  Skipping tile {tile_name}: unexpected geometry type "
                  f"{tile_geom.geom_type} after clipping")
            continue

        if ring[0] != ring[-1]:
            ring.append(ring[0])

        region_name = f"tile_{tile_name}"
        out[region_name] = {
            "type": "Polygon",
            "coordinates": [ring],
        }

        # Aggregate dates from every source region that touches this tile
        all_start, all_end = [], []
        for _, mr in touching.iterrows():
            sd = mr.get("start_dates_right", mr.get("start_dates", [])) or []
            ed = mr.get("end_dates_right",   mr.get("end_dates",   [])) or []
            all_start.extend(sd)
            all_end.extend(ed)

        start_dates_out[region_name] = sorted(set(all_start)) if all_start else []
        end_dates_out[region_name]   = sorted(set(all_end))   if all_end   else []

    print(f"  Resolved {len(out)} Sentinel-2 tile(s) covering {len(selected_keys)} selected region(s)")
    return out, start_dates_out, end_dates_out


# ─────────────────────────────────────────────────────────────────────────────
# Grid loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet(parquet_path: Path) -> gpd.GeoDataFrame:
    """Load a sentinel-2-grid.parquet file (WKB-encoded geometry column)."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkb(df["geometry"]),
        crs="EPSG:4326",
    )
    print(f"    Loaded parquet ({len(gdf)} tiles)")
    return gdf


def _load_kml(kml_path: Path) -> gpd.GeoDataFrame:
    """
    Load an ESA Sentinel-2 KML tile grid using only xml.etree (no fiona/KML driver).

    The ESA KML structure per Placemark:

        <Placemark>
            <name>18TXR</name>
            <MultiGeometry>
                <Polygon>
                    <outerBoundaryIs><LinearRing>
                        <coordinates>lon,lat,z lon,lat,z ...</coordinates>
                    </LinearRing></outerBoundaryIs>
                </Polygon>
                ...   (tiles crossing the antimeridian have 2 Polygons)
            </MultiGeometry>
        </Placemark>

    Z values are dropped (lon, lat only), matching LeafWrapper's drop_z behaviour.
    Multi-polygon tiles (antimeridian-crossing) are kept as MultiPolygon — the
    spatial join in resolve_s2_tiles handles them correctly via geopandas.
    """
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

    KML_NS = "http://www.opengis.net/kml/2.2"

    # Pre-build qualified tag strings once rather than repeating f-strings
    placemark_tag = f"{{{KML_NS}}}Placemark"
    name_tag      = f"{{{KML_NS}}}name"
    polygon_tag   = f"{{{KML_NS}}}Polygon"
    outer_tag     = f"{{{KML_NS}}}outerBoundaryIs"
    ring_tag      = f"{{{KML_NS}}}LinearRing"
    coords_tag    = f"{{{KML_NS}}}coordinates"

    def _parse_coords(coord_text: str):
        """
        Parse a KML <coordinates> text block → list of (lon, lat) tuples.
        KML triples are whitespace-separated: 'lon,lat,z lon,lat,z ...'
        Z is silently dropped — matches LeafWrapper's drop_z behaviour.
        """
        pts = []
        for token in coord_text.split():
            token = token.strip()
            if not token:
                continue
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        return pts

    tree = ET.parse(kml_path)
    root = tree.getroot()

    rows = []
    for pm in root.iter(placemark_tag):
        # ── Tile ID from <name> ────────────────────────────────────────────
        name_el = pm.find(name_tag)
        tile_id = name_el.text.strip() if (name_el is not None and name_el.text) else None
        if not tile_id:
            continue

        # ── Collect all <Polygon> rings inside this Placemark ──────────────
        polys = []
        for poly_el in pm.iter(polygon_tag):
            outer = poly_el.find(outer_tag)
            if outer is None:
                continue
            ring_el = outer.find(ring_tag)
            if ring_el is None:
                continue
            coord_el = ring_el.find(coords_tag)
            if coord_el is None or not coord_el.text:
                continue

            pts = _parse_coords(coord_el.text)
            if len(pts) < 3:
                continue

            # Close the ring if needed 
            if pts[0] != pts[-1]:
                pts.append(pts[0])

            poly = ShapelyPolygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0) 
            if not poly.is_empty:
                polys.append(poly)

        if not polys:
            continue

        # Single polygon → Polygon; two polygons (antimeridian) → MultiPolygon
        geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        rows.append({"Name": tile_id, "geometry": geom})

    if not rows:
        raise ValueError(
            f"No Placemark geometries parsed from KML: {kml_path}\n"
            "Verify this is an ESA Sentinel-2 tile grid KML file."
        )

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    print(f"    Loaded KML via xml.etree ({len(gdf)} tiles)")
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_tile_name_column(gdf: gpd.GeoDataFrame) -> str:
    """Auto-detect the column containing the MGRS tile name in the S2 grid GeoDataFrame."""
    # Ordered preference list of known column names across common S2 grid sources.
    # "Name" is first because that is what _load_kml produces and what the ESA KML
    # uses natively via fiona as well.
    candidates = [
        "Name", "name", "TILE", "tile", "TILE_ID", "tile_id",
        "id", "ID", "mgrs", "MGRS", "tileid",
    ]
    for c in candidates:
        if c in gdf.columns:
            return c

    # Fallback: first non-geometry string column
    for c in gdf.columns:
        if c == "geometry":
            continue
        if gdf[c].dtype == object:
            return c

    raise ValueError(
        f"Cannot find tile name column in S2 grid.\n"
        f"Available columns: {list(gdf.columns)}\n"
        f"Please check which column holds the MGRS tile name (e.g. '18TXR')."
    )