# Change log

## 0.6.0 (27/03/2022)

### Features

- feat: CLI args types, docstrings; fix aoi CLI arg; fix CLI tests
- feat: rm stale config files
- feat: accept aoi_vector_filepath arg
- feat: compare random arg in `get_model_perf_df`
- feat: predict t_da using rioxarray (also dropped dask)

### Fixes

- fix: drop stale dask requirement
- fix: factors/intensity param keys:values in cli dump solution
- fix: rm align CLI option and rm align req for t_raster in docstrings
- fix: test the dates argument
- fix: proper management of the dates arg/attr

### Other

- docs: remove refs to stale `predict_t` methods
- docs: updated license file for github badge
- docs: updated user guide for CLI (quotes, comma/space sep)
- docs: updated README (install instructions, license badge)
- docs: fix path to cli function, rtd install package
- ci: skip existing in github actions test pypi
- ci: added aoi tests
- ci: fix num_steps/num_update_logs in cli tests
- ci: test both cc_methods
- refactor: use sample_comparison_df in calibration energy
- ci: test cli for both comma and space separators
- ci: calibrate in module tests
- refactor: use keyword-only args
- style: using ruff with numpy docstrings
- refactor: using Python 3 super class syntax
- docs: updated RTD config to use conda

## 0.5.0 (07/01/2022)

- feat: run invest to align rasters to ensure consistent raster meta
- fix: data array groupby apply -> map (deprecation)
- feat: CLI using fire instead of click
- fix: rio crs to str for fiona (not needed in fiona>=1.9)

## 0.4.1 (11/09/2020)

- compute r2 score with scipy.stats to avoid negative values

## 0.4.0 (10/09/2020)

- dropped `predict_t` method (replaced by `predict_t_arrs`)
- use sample name, index and keys attributes to index the samples

## 0.3.3 (18/08/2020)

- corrected `obs_arr` and `dates` in `UCMWrapper.__init__` and test that certain functionalities can work without providing observed temperature values

## 0.3.2 (18/08/2020)

- correct docs: math, predict_t_arr, opt. UCMWrapper args
- optional `t_raster_filepaths`/`station_t_filepath` in `UCMWrapper`

## 0.3.1 (28/07/2020)

- fix readthedocs build issues by adding more libraries to `autodoc_mock_imports`
- added docstrings for public API methods
- added `UCMCalibrator.calibrate` method (uses `Annealer.anneal`)
- fix passing `ucm_args` to `get_comparison_df` and `get_model_perf_df` methods

## 0.3.0 (27/07/2020)

- custom `dates` argument in `UCMWrapper`
- shortcut to useful `UCMWrapper` methods in `UCMCalibrator`
- using `_ucm_params_dict` property to get params from annealer state
- added sample comparison and model performance methods
- added `predict_t_da` method (works with xarray)
- renamed variables `model_args` -> `ucm_args`, `DEFAULT_MODEL_PARAMS` -> `DEFAULT_UCM_PARAMS`
- default model parameters from `settings` module in `base_args` attribute of the `UCMWrapper` class

## 0.2.1 (23/06/2020)

- update `base_args` with `model_args` in `predict_t_arr`
- compute R^2 with `scipy.stats` (instead of `sklearn.metrics`)
- exclude zero kernel distance to avoid nan/infinity errors

## 0.2.0 (18/06/2020)

- set `natcap.invest` version requirement as `>=3.8.0`, restrict pygeoprocessing to `<2.0`
- alignment of the temperature rasters if needed
- using `src.dataset_mask()` method instead of `arr != src.nodata`
- dumped `aoi_vector_filepath` argument, automatically generating a dummy one instead (since it is not used in the calibration)

## 0.1.1 (08/05/2020)

- fix automatic defaults for `num_steps` and `num_update_logs`
- set num_steps/num_update_logs to 2 in tests to test `move` method

## 0.1.0 (07/05/2020)

- initial release
