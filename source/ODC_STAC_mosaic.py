from datetime import datetime, timedelta

import xarray as xr
import pystac_client
import stackstac
import odc.stac
from odc.geo.geobox import GeoBox
from dask.diagnostics import ProgressBar
from rasterio.enums import Resampling

dx = 3/3600  # 90m resolution
epsg = 4326
# random location in the middle of the Amazon
bounds = (-64.0, -9.0, -63.5, -8.5)
minx, miny, maxx, maxy = bounds
geom = {
    'type': 'Polygon',
    'coordinates': [[
       [minx, miny],
       [minx, maxy],
       [maxx, maxy],
       [maxx, miny],
       [minx, miny]
    ]]
}


year = 2020
month = 1

start_date = datetime(year, month, 1)
end_date = start_date + timedelta(days=31)
date_query = start_date.strftime("%Y-%m-%d") + "/" + end_date.strftime("%Y-%m-%d")


items = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")\
        .search(intersects=geom, collections=["sentinel-2-l2a"], datetime=date_query, limit=100)\
        .item_collection()

print(len(items), "scenes found")

# define a geobox for my region
geobox = GeoBox.from_bbox(bounds, crs=f"epsg:{epsg}", resolution=dx)

# lazily combine items
ds_odc = odc.stac.load(
    items,
    bands=["scl", "red", "green", "blue"],
    chunks={'time': 5, 'x': 600, 'y': 600},
    geobox=geobox,
    resampling="bilinear")

# actually load it
with ProgressBar():
    ds_odc.load()

# define a mask for valid pixels (non-cloud)
def is_valid_pixel(data):
    # include only vegetated, not_vegitated, water, and snow
    return ((data > 3) & (data < 7)) | (data==11)

ds_odc['valid'] = is_valid_pixel(ds_odc.scl)
ds_odc.valid.sum("time").plot()

# compute the masked median
rgb_median = (
    ds_odc[['red', 'green', 'blue']]
    .where(ds_odc.valid)
    .to_dataarray(dim="band")
    .transpose(..., "band")
    .median(dim="time")
)
(rgb_median / rgb_median.max() * 2).plot.imshow(rgb="band", figsize=(10, 8))