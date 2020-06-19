# Change log

## 0.2.0 (18/06/2020)

* set `natcap.invest` version requirement as `>=3.8.0`, restrict pygeoprocessing to `<2.0`
* alignment of the temperature rasters if needed
* using `src.dataset_mask()` method instead of `arr != src.nodata`
* dumped `aoi_vector_filepath` argument, automatically generating a dummy one instead (since it is not used in the calibration)

## 0.1.1 (08/05/2020)

* fix automatic defaults for `num_steps` and `num_update_logs`
* set num_steps/num_update_logs to 2 in tests to test `move` method

## 0.1.0 (07/05/2020)

* initial release
