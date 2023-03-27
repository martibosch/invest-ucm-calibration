import os
import string
from os import path

import fiona
import numpy as np
import osr
import pandas as pd
import rasterio as rio
from numpy import random as rn
from rasterio import features, transform


def dump_rasters(raster_filepaths, low, high, nodata, data_mask, dtype, meta):
    for raster_filepath in raster_filepaths:
        with rio.open(raster_filepath, "w", dtype=dtype, nodata=nodata, **meta) as dst:
            dst_arr = (
                rn.uniform(low, high, meta["height"] * meta["width"])
                .reshape(meta["height"], meta["width"])
                .astype(dtype)
            )
            dst.write(np.where(data_mask, dst_arr, nodata), 1)


tests_data_dir = "tests/data"

if not path.exists(tests_data_dir):
    os.mkdir(tests_data_dir)

# basic config
width, height = 5, 5
num_dates = 2
shape = (height, width)
num_pixels = width * height
west, south, east, north = 0, 0, 10, 10
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
meta = {
    "driver": "GTiff",
    "width": width,
    "height": height,
    "count": 1,
    "transform": transform.from_bounds(west, south, east, north, width, height),
    "crs": srs.ExportToProj4(),
}

data_mask = np.full(shape, True)
data_mask[0, 0] = False

lulc_raster_filepath = path.join(tests_data_dir, "lulc.tif")
num_lulc_classes = 4
lulc_nodata = 255
lulc_dtype = np.uint8

biophysical_table_filepath = path.join(tests_data_dir, "biophysical-table.csv")

# aoi_vector_filepath = path.join(tests_data_dir, 'aoi-vector.shp')

ref_et_raster_filepaths = [
    path.join(tests_data_dir, f"ref_et{i}.tif") for i in range(num_dates)
]
ref_et_low = 1
ref_et_high = 3
ref_et_nodata = -1
ref_et_dtype = np.float32

t_raster_filepaths = [path.join(tests_data_dir, f"T{i}.tif") for i in range(num_dates)]
unaligned_t_raster_filepaths = [
    path.join(tests_data_dir, f"_T{i}.tif") for i in range(num_dates)
]
t_low = 20
t_high = 30
t_nodata = -300
t_dtype = np.float32

num_stations = 2
station_labels = list(string.ascii_lowercase[:num_stations])

# dump a randomly-generated lulc raster
with rio.open(
    lulc_raster_filepath, "w", dtype=lulc_dtype, nodata=lulc_nodata, **meta
) as dst:
    dst_arr = np.digitize(
        rn.random(num_pixels).reshape(shape), np.linspace(0, 1, 4)
    ).astype(lulc_dtype)
    dst.write(np.where(data_mask, dst_arr, lulc_nodata), 1)

# dump a randomly-generated biophysical table
biophysical_df = pd.DataFrame()
biophysical_df["lucode"] = np.arange(1, num_lulc_classes + 1)
for column in "kc", "albedo", "shade":
    biophysical_df[column] = rn.random(num_lulc_classes)
biophysical_df["green_area"] = rn.choice([0, 1], num_lulc_classes)
biophysical_df.to_csv(biophysical_table_filepath)

# # dump an aoi vector shapefile with a single geometry (bounding box)
# # based on Mike T answer in stackoverflow https://bit.ly/3dp7Ihb
# # Define a polygon feature geometry with one attribute
# schema = {
#     'geometry': 'Polygon',
#     'properties': {
#         'id': 'int'
#     },
# }
# with fiona.open(aoi_vector_filepath, mode='w', driver='ESRI Shapefile',
#                 schema=schema, crs=meta['crs']) as dst:
#     ## If there are multiple geometries, put the "for" loop here
#     dst.write({
#         'geometry':
#         geometry.mapping(geometry.box(west, south, east, north)),
#         'properties': {
#             'id': 123
#         },
#     })

# dump a randomly-generated ref et raster
dump_rasters(
    ref_et_raster_filepaths,
    ref_et_low,
    ref_et_high,
    ref_et_nodata,
    data_mask,
    ref_et_dtype,
    meta,
)

# dump a randomly-generated t raster
dump_rasters(t_raster_filepaths, t_low, t_high, t_nodata, data_mask, t_dtype, meta)

# dump a randomly-generated unaligned t raster
unaligned_meta = meta.copy()
unaligned_shape = height - 1, width - 1
unaligned_meta.update(width=width - 1, height=height - 1)
dump_rasters(
    unaligned_t_raster_filepaths,
    t_low,
    t_high,
    t_nodata,
    data_mask[1:, 1:],
    t_dtype,
    unaligned_meta,
)

# dump a randomly-generated station location data frame
station_location_df = pd.DataFrame(index=station_labels)
station_location_df["x"] = rn.uniform(west, east, num_stations)
station_location_df["y"] = rn.uniform(south, north, num_stations)
station_location_df.to_csv(path.join(tests_data_dir, "station-locations.csv"))

# dump the randomly-generated station measurements data frames
station_t_df = pd.DataFrame(columns=station_labels)
station_t_df.loc[0] = rn.uniform(t_low, t_high, num_stations)
# one date only
station_t_df.to_csv(path.join(tests_data_dir, "station-t-one-day.csv"))
# two dates
station_t_df.loc[1] = rn.uniform(t_low, t_high, num_stations)
station_t_df.to_csv(path.join(tests_data_dir, "station-t.csv"))

# dump a randomly-generated AOI vector file
with rio.open(lulc_raster_filepath) as src:
    arr = np.ones(src.shape, dtype=rio.int16)
    mask = np.zeros_like(arr).astype(bool)
    mask[1:-1, 1:-1] = True
    shapes = features.shapes(arr, mask=mask, transform=src.transform)

schema = {"geometry": "Polygon", "properties": {"id": "int"}}

# Create a new layer in the GeoPackage file
with fiona.open(
    path.join(tests_data_dir, "aoi.gpkg"), "w", driver="GPKG", schema=schema
) as dst:
    # Write the shapes to the new layer
    for i, (geometry, _value) in enumerate(shapes):
        feature = {"geometry": geometry, "properties": {"id": i + 1}}
        dst.write(feature)
