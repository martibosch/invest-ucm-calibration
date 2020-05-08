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

from . import settings

__version__ = '0.1.1'


# utils
def _is_sequence(arg):
    # # Based on steveha's answer in stackoverflow https://bit.ly/3dpnf0m
    # return (not hasattr(arg, "strip") and hasattr(arg, "__getitem__")
    #         or hasattr(arg, "__iter__"))
    return hasattr(arg, '__getitem__') or hasattr(arg, '__iter__')


def _preprocess_t_rasters(t_raster_filepaths):
    obs_arrs = []
    t_refs = []
    uhi_maxs = []
    for t_raster_filepath in t_raster_filepaths:
        with rio.open(t_raster_filepath) as src:
            t_arr = src.read(1)
            # use `np.nan` for nodata values to ensure that we get the right
            # min/max values with `np.nanmin`/`np.nanmax`
            t_arr = np.where(t_arr != src.nodata, t_arr, np.nan)
            obs_arrs.append(t_arr)
            t_min = np.nanmin(t_arr)
            t_refs.append(t_min)
            uhi_maxs.append(np.nanmax(t_arr) - t_min)

    return np.concatenate(obs_arrs), t_refs, uhi_maxs


def _inverted_r2_score(*args, **kwargs):
    # since we need to maximize (instead of minimize) the r2, the
    # simulated annealing will actually minimize 1 - R^2
    return 1 - metrics.r2_score(*args, **kwargs)


class UCMWrapper:
    def __init__(self, lulc_raster_filepath, biophysical_table_filepath,
                 aoi_vector_filepath, cc_method, ref_et_raster_filepaths,
                 t_refs=None, uhi_maxs=None, t_raster_filepaths=None,
                 station_t_filepath=None, station_locations_filepath=None,
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
            extra_ucm_args = settings.DEFAULT_EXTRA_UCM_ARGS
        if 'do_valuation' not in extra_ucm_args:
            extra_ucm_args['do_valuation'] = settings.DEFAULT_EXTRA_UCM_ARGS[
                'do_valuation']
        self.base_args.update(**extra_ucm_args)

        if workspace_dir is None:
            # TODO: how do we ensure that this is removed?
            workspace_dir = tempfile.mkdtemp()
            # TODO: log to warn that we are using a temporary directory
        # self.workspace_dir = workspace_dir
        self.base_args.update(workspace_dir=workspace_dir)

        # evapotranspiration rasters for each date
        if isinstance(ref_et_raster_filepaths, str):
            ref_et_raster_filepaths = [ref_et_raster_filepaths]
        self.ref_et_raster_filepaths = ref_et_raster_filepaths

        # calibration approaches
        if t_raster_filepaths is not None:
            # calibrate against a map
            if isinstance(t_raster_filepaths, str):
                t_raster_filepaths = [t_raster_filepaths]

            # Tref and UHImax
            if t_refs is None:
                if uhi_maxs is None:
                    obs_arr, t_refs, uhi_maxs = _preprocess_t_rasters(
                        t_raster_filepaths)
                else:
                    obs_arr, t_refs, _ = _preprocess_t_rasters(
                        t_raster_filepaths)
            else:
                if uhi_maxs is None:
                    obs_arr, _, uhi_maxs = _preprocess_t_rasters(
                        t_raster_filepaths)
                else:
                    obs_arr, _, __ = _preprocess_t_rasters(t_raster_filepaths)

            # method to predict the temperature values
            self._predict_t = self.predict_t_arr
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

            station_t_df = pd.read_csv(station_t_filepath,
                                       index_col=0)[station_location_df.index]
            station_t_df.index = pd.to_datetime(station_t_df.index)
            self.dates = station_t_df.index
            self.station_tair_df = station_t_df

            # tref and uhi max
            if t_refs is None:
                t_refs = station_t_df.min(axis=1)
            if uhi_maxs is None:
                uhi_maxs = station_t_df.max(axis=1) - t_refs

            # prepare the observation array
            obs_arr = station_t_df.values  # .flatten()

            # method to predict the temperature values
            self._predict_t = self._predict_t_stations

        # store reference temperatures and UHI magnitudes as class attributes
        if not _is_sequence(t_refs):
            t_refs = [t_refs]
        if not _is_sequence(uhi_maxs):
            uhi_maxs = [uhi_maxs]
        self.t_refs = t_refs
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

    def predict_t_arr(self, i, model_args=None):
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
            t_ref=self.t_refs[i],
            uhi_max=self.uhi_maxs[i])
        ucm.execute(date_args)

        with rio.open(
                path.join(date_args['workspace_dir'], 'intermediate',
                          'T_air.tif')) as src:
            # return src.read(1, **read_kws)
            return src.read(1)

    def _predict_t_stations(self, i, model_args=None):
        return self.predict_t_arr(
            i, model_args)[self.station_rows, self.station_cols]

    def predict_t(self, model_args=None):
        # we could also iterate over `self.t_refs` or `self.uhi_maxs`
        pred_delayed = [
            dask.delayed(self._predict_t)(i, model_args)
            for i in range(len(self.ref_et_raster_filepaths))
        ]

        return np.hstack(
            dask.compute(*pred_delayed, scheduler='processes',
                         num_workers=self.num_workers))

    # TODO: support unaligned rasters?
    # def _predict_t_map(self, date, model_args=None):
    #     return self.predict_t_arr(date, model_args, read_kws={
    #         'out_shape': self.map_shape,
    #         'resampling': self.resampling
    #     }).flatten()


class UCMCalibrator(simanneal.Annealer):
    def __init__(self, lulc_raster_filepath, biophysical_table_filepath,
                 aoi_vector_filepath, cc_method, ref_et_raster_filepaths,
                 t_refs=None, uhi_maxs=None, t_raster_filepaths=None,
                 station_t_filepath=None, station_locations_filepath=None,
                 workspace_dir=None, initial_solution=None,
                 extra_ucm_args=None, metric=None, stepsize=None,
                 num_workers=None, num_steps=None, num_update_logs=None):
        """
        Parameters
        ----------
        lulc_raster_filepath : str
            Path to the raster of land use/land cover (LULC) file
        biophysical_table_filepath : str
            Path to the biophysical table CSV file
        aoi_vector_filepath : str
            Path to a vector delineating the areas of interest (required to
            launch the urban cooling model, but it does not affect the
            calibration)
        cc_method : str
            Cooling capacity calculation method. Can be either 'factors' or
            'intensity'
        ref_et_raster_filepaths : str or list-like
            Path to the reference evapotranspiration raster, or sequence of
            strings with a path to the reference evapotranspiration raster
        t_refs : numeric or list-like, optional
            Reference air temperature. If not provided, it will be set as the
            minimum observed temperature (raster or station measurements, for
            each respective date if calibrating for multiple dates).
        uhi_maxs : numeric or list-like, optional
            Magnitude of the UHI effect. If not provided, it will be set as the
            difference between the maximum and minimum observed temperature
            (raster or station measurements, for each respective date if
            calibrating for multiple dates).
        t_raster_filepaths : str or list-like, optional
            Path to the observed temperature raster, or sequence of strings
            with a path to the observed temperature rasters. The raster must
            be aligned to the LULC raster. Required if calibrating against
            temperature map(s).
        station_t_filepath : str, optional
            Path to a table of air temperature measurements where each column
            corresponds to a monitoring station and each row to a datetime.
            Required if calibrating against station measurements.
        station_locations_filepath : str, optional
            Path to a table with the locations of each monitoring station,
            where the first column features the station labels (that match the
            columns of the table of air temperature measurements), and there
            are (at least) a column labelled 'x' and a column labelled 'y'
            that correspod to the locations of each station (in the same CRS
            as the other rasters). Required if calibrating against station
            measurements.
        workspace_dir : str, optional
            Path to the folder where the model outputs will be written. If not
            provided, a temporary directory will be used.
        initial_solution : list-like, optional
            Sequence with the parameter values used as initial solution, of
            the form (t_air_average_radius, green_area_cooling_distance,
            cc_weight_shade, cc_weight_albedo, cc_weight_eti). If not provided,
            the default values of the urban cooling model will be used.
        extra_ucm_args : dict-like, optional
            Other keyword arguments to be passed to the `execute` method of
            the urban cooling model.
        metric : {'R2', 'MAE', 'RMSE'}, optional
            Target metric to optimize in the calibration. Can be either 'R2'
            for the R squared (which will be maximized), 'MAE' for the mean
            absolute error (which will be minimized) or 'RMSE' for the (root)
            mean squared error (which will be minimized). If not provided, the
            value set in `settings.DEFAULT_METRIC` will be used.
        stepsize : numeric, optional
            Step size in terms of the fraction of each parameter when looking
            to select a neighbor solution for the following iteration. The
            neighbor will be randomly drawn from an uniform distribution in the
            [param - stepsize * param, param + stepsize * param] range. For
            example, with a step size of 0.3 and a 't_air_average_radius' of
            500 at a given iteration, the solution for the next iteration will
            be uniformly sampled from the [350, 650] range. If not provided, it
            will be taken from `settings.DEFAULT_STEPSIZE`.
        num_workers : int, optional
            Number of workers so that the simulations of each iteration can be
            executed at scale. Only useful if calibrating for multiple dates.
            If not provided, it will be set automatically depending on the
            number of dates and available number of processors in the CPU.
        num_steps : int, optional.
            Number of iterations of the simulated annealing procedure. If not
            provided, the value set in `settings.DEFAULT_NUM_STEPS` will be
            used.
        num_update_logs : int, default 100
            Number of updates that will be logged. If `num_steps` is equal to
            `num_update_logs`, each iteration will be logged. If not provided,
            the value set in `settings.DEFAULT_UPDATE_LOGS` will be used.
        """
        # init the model wrapper
        self.ucm_wrapper = UCMWrapper(
            lulc_raster_filepath, biophysical_table_filepath,
            aoi_vector_filepath, cc_method, ref_et_raster_filepaths,
            t_refs=t_refs, uhi_maxs=uhi_maxs,
            t_raster_filepaths=t_raster_filepaths,
            station_t_filepath=station_t_filepath,
            station_locations_filepath=station_locations_filepath,
            workspace_dir=workspace_dir, extra_ucm_args=extra_ucm_args,
            num_workers=num_workers)

        # metric
        if metric is None:
            metric = settings.DEFAULT_METRIC
        if metric == 'R2':
            # since we need to maximize (instead of minimize) the r2, the
            # simulated annealing will actually minimize 1 - R^2
            self.compute_metric = _inverted_r2_score
        elif metric == 'MAE':
            self.compute_metric = metrics.mean_absolute_error
        else:  # 'RMSE'
            self.compute_metric = metrics.mean_squared_error

        # step size to find neigbhor solution
        if stepsize is None:
            stepsize = settings.DEFAULT_STEPSIZE
        self.stepsize = stepsize

        # initial solution
        if initial_solution is None:
            initial_solution = list(settings.DEFAULT_MODEL_PARAMS.values())
        # init the parent `Annealer` instance with the initial solution
        super(UCMCalibrator, self).__init__(initial_solution)

        # nicer parameters for the urban cooling model solution space
        if num_steps is None:
            num_steps = settings.DEFAULT_NUM_STEPS
        self.steps = num_steps
        if num_update_logs is None:
            num_update_logs = settings.DEFAULT_NUM_UPDATE_LOGS
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
        model_args = self.ucm_wrapper.base_args.copy()
        model_args.update(t_air_average_radius=self.state[0],
                          green_area_cooling_distance=self.state[1],
                          cc_weight_shade=self.state[2],
                          cc_weight_albedo=self.state[3],
                          cc_weight_eti=self.state[4])

        pred_arr = self.ucm_wrapper.predict_t(model_args=model_args).flatten()

        return self.compute_metric(self.ucm_wrapper.obs_arr,
                                   pred_arr[self.ucm_wrapper.obs_mask])
