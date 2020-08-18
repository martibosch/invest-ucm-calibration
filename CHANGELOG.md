# Change log

## 0.3.3 (18/08/2020)

* corrected `obs_arr` and `dates` in `UCMWrapper.__init__` and test that certain functionalities can work without providing observed temperature values

## 0.3.2 (18/08/2020)

* correct docs: math, predict_t_arr, opt. UCMWrapper args
* optional `t_raster_filepaths`/`station_t_filepath` in `UCMWrapper`

## 0.3.1 (28/07/2020)

* fix readthedocs build issues by adding more libraries to `autodoc_mock_imports`
* added docstrings for public API methods
* added `UCMCalibrator.calibrate` method (uses `Annealer.anneal`)
* fix passing `ucm_args` to `get_comparison_df` and `get_model_perf_df` methods

## 0.3.0 (27/07/2020)
 
* custom `dates` argument in `UCMWrapper`
* shortcut to useful `UCMWrapper` methods in `UCMCalibrator`
* using `_ucm_params_dict` property to get params from annealer state
* added sample comparison and model performance methods
* added `predict_t_da` method (works with xarray) 
* renamed variables `model_args` -> `ucm_args`, `DEFAULT_MODEL_PARAMS` -> `DEFAULT_UCM_PARAMS`
* default model parameters from `settings` module in `base_args` attribute of the `UCMWrapper` class

## 0.2.1 (23/06/2020)

* update `base_args` with `model_args` in `predict_t_arr`
* compute R^2 with `scipy.stats` (instead of `sklearn.metrics`)
* exclude zero kernel distance to avoid nan/infinity errors

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
