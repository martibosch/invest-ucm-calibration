import os
import tempfile
from os import path

import dask
import numpy as np
import numpy.random as rn
import pandas as pd
import rasterio as rio
import simanneal
from natcap.invest import urban_cooling_model as ucm
from rasterio import transform
from sklearn import metrics

__version__ = '0.0.1'

# constants useful for the invest ucm
# DEFAULT_INITIAL_SOLUTION = [500, 100, 0.6, 0.2, 0.2]
DEFAULT_MODEL_PARAMS = {
    't_air_average_radius': 500,
    'green_area_cooling_distance': 100,
    'cc_weight_shade': 0.6,
    'cc_weight_albedo': 0.2,
    'cc_weight_eti': 0.2
}
DEFAULT_EXTRA_UCM_ARGS = {'do_valuation': False}


# utils
def _is_sequence(arg):
    # Based on steveha's answer in stackoverflow https://bit.ly/3dpnf0m
    return (not hasattr(arg, "strip") and hasattr(arg, "__getitem__")
            or hasattr(arg, "__iter__"))


def _preprocess_T_rasters(T_raster_filepaths):
    obs_arrs = []
    T_refs = []
    uhi_maxs = []
    for T_raster_filepath in T_raster_filepaths:
        with rio.open(T_raster_filepath) as src:
            T_arr = src.read(1)
            # use `np.nan` for nodata values to ensure that we get the right
            # min/max values with `np.nanmin`/`np.nanmax`
            T_arr = np.where(T_arr == src.nodata, T_arr, np.nan)
            obs_arrs.append(T_arr)
            T_min = np.nanmin(T_arr)
            T_refs.append(T_min)
            uhi_maxs.append(np.nanmax(T_arr) - T_min)
    return np.concatenate(obs_arrs), T_refs, uhi_maxs


def _inverted_r2_score(*args, **kwargs):
    # since we need to maximize (instead of minimize) the r2, the
    # simulated annealing will actually minimize 1 - R^2
    return 1 - metrics.r2_score(*args, **kwargs)


class ModelWrapper:
    def __init__(self, lulc_raster_filepath, biophysical_table_filepath,
                 aoi_vector_filepath, cc_method, ref_et_raster_filepaths,
                 T_refs=None, uhi_maxs=None, T_raster_filepaths=None,
                 station_T_filepath=None, station_locations_filepath=None,
                 workspace_dir=None, extra_ucm_args=None, num_workers=None):
        # model parameters
        self.base_args = {
            'lulc_raster_path': lulc_raster_filepath,
            'biophysical_table_path': biophysical_table_filepath,
            'aoi_vector_path': aoi_vector_filepath,
            'cc_method': cc_method,
        }
        # if model_params is None:
        #     model_params = DEFAULT_MODEL_PARAMS
        # self.base_args.update(**model_params)
        if extra_ucm_args is None:
            extra_ucm_args = DEFAULT_EXTRA_UCM_ARGS
        if 'do_valuation' not in extra_ucm_args:
            extra_ucm_args['do_valuation'] = DEFAULT_EXTRA_UCM_ARGS[
                'do_valuation']
        self.base_args.update(**extra_ucm_args)

        if workspace_dir is None:
            # TODO: how do we ensure that this is removed?
            workspace_dir = tempfile.mkdtemp()
            # TODO: log to warn that we are using a temporary directory
        # self.workspace_dir = workspace_dir
        self.base_args.update(workspace_dir=workspace_dir)

        # evapotranspiration rasters for each date
        if not _is_sequence(ref_et_raster_filepaths):
            ref_et_raster_filepaths = [ref_et_raster_filepaths]
        self.ref_et_raster_filepaths = ref_et_raster_filepaths

        # calibration approaches
        if T_raster_filepaths is not None:
            # calibrate against a map
            if not _is_sequence(T_raster_filepaths):
                T_raster_filepaths = [T_raster_filepaths]
            # Tref and UHImax
            if T_refs is None:
                if uhi_maxs is None:
                    obs_arr, T_refs, uhi_maxs = _preprocess_T_rasters(
                        T_raster_filepaths)
                else:
                    obs_arr, T_refs, _ = _preprocess_T_rasters(
                        T_raster_filepaths)
            else:
                if uhi_maxs is None:
                    obs_arr, _, uhi_maxs = _preprocess_T_rasters(
                        T_raster_filepaths)
                else:
                    obs_arr, _, __ = _preprocess_T_rasters(T_raster_filepaths)

            # method to predict the temperature values
            self._predict_T = self.predict_T_arr
            # TODO: use xarray?
            # T_da = xr.open_dataarray(tair_da_filepath)
            # self.Tref_ser = T_da.groupby('time').min(['x', 'y']).to_pandas()
            # self.uhi_max_ser = T_da.groupby('time').max(
            #     ['x', 'y']).to_pandas() - self.Tref_ser
            # prepare the flat observation array
            # obs_arr = T_da.values.flatten()

            # TODO: support unaligned rasters?
            # # prepare the cost function and its arguments
            # # shape of the map (for each date)
            # self.map_shape = T_da.shape[1:]
            # self.resampling = Resampling.bilinear
        else:
            station_location_df = pd.read_csv(station_locations_filepath,
                                              index_col=0)
            with rio.open(lulc_raster_filepath) as src:
                self.station_rows, self.station_cols = transform.rowcol(
                    src.transform, station_location_df['x'],
                    station_location_df['y'])
                # useful to predict air temperature rasters
                self.meta = src.meta.copy()
                self.data_mask = src.dataset_mask().astype(bool)

            station_T_df = pd.read_csv(station_T_filepath,
                                       index_col=0)[station_location_df.index]
            station_T_df.index = pd.to_datetime(station_T_df.index)
            self.dates = station_T_df.index
            self.station_tair_df = station_T_df

            # tref and uhi max
            if T_refs is None:
                T_refs = station_T_df.min(axis=1)
            if uhi_maxs is None:
                uhi_maxs = station_T_df.max(axis=1) - T_refs

            # prepare the observation array
            obs_arr = station_T_df.values  # .flatten()

            # method to predict the temperature values
            self._predict_T = self._predict_T_stations

        # store reference temperatures and UHI magnitudes as class attributes
        if not _is_sequence(T_refs):
            T_refs = [T_refs]
        if not _is_sequence(uhi_maxs):
            uhi_maxs = [uhi_maxs]
        self.T_refs = T_refs
        self.uhi_maxs = uhi_maxs

        # flat observation array to compute the calibration metric
        self.obs_arr = obs_arr.flatten()
        self.obs_mask = ~np.isnan(self.obs_arr)
        self.obs_arr = self.obs_arr[self.obs_mask]

        # number of workers to perform each calibration iteration at scale
        if num_workers is None:
            num_workers = min(len(self.ref_et_raster_filepaths),
                              os.cpu_count())
        self.num_workers = num_workers

    def predict_T_arr(self, i, model_args=None):
        if model_args is None:
            model_args = self.base_args
        # TODO: support unaligned rasters?
        # if read_kws is None:
        #     read_kws = {}

        # note that this workspace_dir corresponds to this date only
        workspace_dir = path.join(self.base_args['workspace_dir'], str(i))
        date_args = model_args.copy()
        date_args.update(
            workspace_dir=workspace_dir,
            ref_eto_raster_path=self.ref_et_raster_filepaths[i],
            # t_ref=Tref_da.sel(time=date).item(),
            # uhi_max=uhi_max_da.sel(time=date).item()
            t_ref=self.T_refs[i],
            uhi_max=self.uhi_maxs[i])
        ucm.execute(date_args)

        with rio.open(
                path.join(date_args['workspace_dir'], 'intermediate',
                          'T_air.tif')) as src:
            # return src.read(1, **read_kws)
            return src.read(1)

    def _predict_T_stations(self, i, model_args=None):
        return self.predict_T_arr(
            i, model_args)[self.station_rows, self.station_cols]

    def predict_T(self, model_args=None):
        # we could also iterate over `self.T_refs` or `self.uhi_maxs`
        pred_delayed = [
            dask.delayed(self._predict_T)(i, model_args)
            for i in range(len(self.ref_et_raster_filepaths))
        ]

        return np.hstack(
            dask.compute(*pred_delayed, scheduler='processes',
                         num_workers=self.num_workers))

    # TODO: support unaligned rasters?
    # def _predict_T_map(self, date, model_args=None):
    #     return self.predict_T_arr(date, model_args, read_kws={
    #         'out_shape': self.map_shape,
    #         'resampling': self.resampling
    #     }).flatten()


class UCMCalibrator(simanneal.Annealer):
    def __init__(self, lulc_raster_filepath, biophysical_table_filepath,
                 aoi_vector_filepath, cc_method, ref_et_raster_filepaths,
                 T_refs=None, uhi_maxs=None, T_raster_filepaths=None,
                 station_T_filepath=None, station_locations_filepath=None,
                 workspace_dir=None, initial_solution=None,
                 extra_ucm_args=None, metric='R2', stepsize=0.3,
                 num_workers=None, num_steps=100, num_update_logs=100):
        self.mw = ModelWrapper(
            lulc_raster_filepath, biophysical_table_filepath,
            aoi_vector_filepath, cc_method, ref_et_raster_filepaths,
            T_refs=T_refs, uhi_maxs=uhi_maxs,
            T_raster_filepaths=T_raster_filepaths,
            station_T_filepath=station_T_filepath,
            station_locations_filepath=station_locations_filepath,
            workspace_dir=workspace_dir, extra_ucm_args=extra_ucm_args,
            num_workers=num_workers)
        # metric
        if metric == 'R2':
            # since we need to maximize (instead of minimize) the r2, the
            # simulated annealing will actually minimize 1 - R^2
            self.compute_metric = _inverted_r2_score
        elif metric == 'MAE':
            self.compute_metric = metrics.mean_absolute_error
        else:  # 'RMSE'
            self.compute_metric = metrics.mean_squared_error
        # step size to find neigbhor solution
        self.stepsize = stepsize
        # initial solution
        if initial_solution is None:
            initial_solution = list(DEFAULT_MODEL_PARAMS.values())
        # init the parent `Annealer` instance with the initial solution
        super(UCMCalibrator, self).__init__(initial_solution)
        # nicer parameters for the urban cooling model solution space
        self.steps = num_steps
        self.updates = num_update_logs

    def move(self):
        state_neighbour = []
        for param in self.state:
            state_neighbour.append(
                param * (1 + rn.uniform(-self.stepsize, self.stepsize)))
        # rescale so that the three weights add up to one
        weight_sum = sum(state_neighbour[2:])
        for k in range(2, 5):
            state_neighbour[k] /= weight_sum

        # update the state
        self.state = state_neighbour

    def energy(self):
        model_args = self.mw.base_args.copy()
        model_args.update(t_air_average_radius=self.state[0],
                          green_area_cooling_distance=self.state[1],
                          cc_weight_shade=self.state[2],
                          cc_weight_albedo=self.state[3],
                          cc_weight_eti=self.state[4])

        pred_arr = self.mw.predict_T(model_args=model_args)

        return self.compute_metric(self.mw.obs_arr, pred_arr[self.mw.obs_mask])
