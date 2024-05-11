import stackstac
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.windows import Window

def calculate_ndvi(red_band, nir_band):
    """Calculate NDVI."""
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi

def composite_ndvi(image_collection):
    """Generate NDVI composite image."""
    red_band = image_collection.select_bands("red")
    nir_band = image_collection.select_bands("nir")

    ndvi = calculate_ndvi(red_band, nir_band)

    return ndvi

def main():
    # Define your area of interest and time range
    bbox = [-180, -90, 180, 90]  # Full global extent
    time_range = ("2023-01-01", "2023-12-31")

    # StackStac Query
    catalog = stackstac.stack("https://earth-search.aws.element84.com/v0")

    # Search for Landsat 8 data
    results = catalog.search(
        bbox=bbox,
        datetime=time_range,
        collections=["landsat-8-l1"],
        query={"eo:cloud_cover": {"lt": 10}},  # Filter by cloud cover
    )

    # Stack the STAC items
    stack = results.stack(assets=["red", "nir"])

    # Generate NDVI composite
    ndvi_composite = composite_ndvi(stack)

    # Plot the NDVI composite
    show(ndvi_composite, cmap="RdYlGn")
    '''
    plt.title("NDVI Composite Image")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    '''
if __name__ == "__main__":
    main()
