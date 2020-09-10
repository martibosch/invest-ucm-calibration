import os
import tempfile
from os import path

import dask
import fiona
import numpy as np
import numpy.random as rn
import pandas as pd
import pygeoprocessing
import rasterio as rio
import simanneal
import xarray as xr
from natcap.invest import urban_cooling_model as ucm
from rasterio import transform
from scipy import stats
from shapely import geometry
from sklearn import metrics

from . import settings

__version__ = '0.4.0'


# utils
def _is_sequence(arg):
    # # Based on steveha's answer in stackoverflow https://bit.ly/3dpnf0m
    # return (not hasattr(arg, "strip") and hasattr(arg, "__getitem__")
    #         or hasattr(arg, "__iter__"))
    return hasattr(arg, '__getitem__') or hasattr(arg, '__iter__')


def _align_rasters(lulc_raster_filepath, ref_et_raster_filepaths,
                   t_raster_filepaths, dst_lulc_raster_filepath,
                   dst_ref_et_raster_filepaths, dst_t_raster_filepaths):
    with rio.open(lulc_raster_filepath) as src:
        pygeoprocessing.align_and_resize_raster_stack(
            [lulc_raster_filepath] + ref_et_raster_filepaths +
            t_raster_filepaths, [dst_lulc_raster_filepath] +
            dst_ref_et_raster_filepaths + dst_t_raster_filepaths,
            ['near'] + ['bilinear'] *
            (len(ref_et_raster_filepaths) + len(t_raster_filepaths)), src.res,
            'intersection')

    # get the intersection mask
    with rio.open(dst_lulc_raster_filepath) as src:
        meta = src.meta.copy()
        data_mask = src.dataset_mask()
    for dst_raster_filepath in dst_ref_et_raster_filepaths + \
            dst_t_raster_filepaths:
        with rio.open(dst_raster_filepath) as src:
            data_mask &= src.dataset_mask()

    for dst_raster_filepath in [dst_lulc_raster_filepath] + \
            dst_ref_et_raster_filepaths + dst_t_raster_filepaths:
        with rio.open(dst_raster_filepath, 'r+') as ds:
            ds.write(
                np.where(data_mask, ds.read(1),
                         ds.nodata).astype(ds.dtypes[0]), 1)

    return (meta, data_mask.astype(bool), dst_lulc_raster_filepath,
            dst_ref_et_raster_filepaths, dst_t_raster_filepaths)


def _preprocess_t_rasters(t_raster_filepaths):
    obs_arrs = []
    t_refs = []
    uhi_maxs = []
    for t_raster_filepath in t_raster_filepaths:
        with rio.open(t_raster_filepath) as src:
            t_arr = src.read(1)
            # use `np.nan` for nodata values to ensure that we get the right
            # min/max values with `np.nanmin`/`np.nanmax`
            t_arr = np.where(src.dataset_mask(), t_arr, np.nan)
            obs_arrs.append(t_arr)
            t_min = np.nanmin(t_arr)
            t_refs.append(t_min)
            uhi_maxs.append(np.nanmax(t_arr) - t_min)

    return obs_arrs, t_refs, uhi_maxs


def _inverted_r2_score(obs, pred):
    # since we need to maximize (instead of minimize) the r2, the
    # simulated annealing will actually minimize 1 - R^2
    slope, intercept, r_value, p_value, std_err = stats.linregress(obs, pred)
    return 1 - r_value * r_value


METRIC_COLUMNS = ['R^2', 'MAE', 'RMSE']


def _compute_model_perf(obs, pred):
    return [
        metrics.r2_score(obs, pred),
        metrics.mean_absolute_error(obs, pred),
        metrics.mean_squared_error(obs, pred, squared=False),
    ]


# classes
class UCMWrapper:
    def __init__(self, lulc_raster_filepath, biophysical_table_filepath,
                 cc_method, ref_et_raster_filepaths, t_refs=None,
                 uhi_maxs=None, t_raster_filepaths=None,
                 station_t_filepath=None, station_locations_filepath=None,
                 dates=None, align_rasters=True, workspace_dir=None,
                 extra_ucm_args=None, num_workers=None):
        """
        Pythonic and open source interface to the InVEST urban cooling model.
        A set of additional utility methods serve to compute temperature maps
        and data frames.

        Parameters
        ----------
        lulc_raster_filepath : str
            Path to the raster of land use/land cover (LULC) file
        biophysical_table_filepath : str
            Path to the biophysical table CSV file
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
            Required if calibrating against station measurements. Ignored if
            providing `t_raster_filepaths`.
        station_locations_filepath : str, optional
            Path to a table with the locations of each monitoring station,
            where the first column features the station labels (that match the
            columns of the table of air temperature measurements), and there
            are (at least) a column labelled 'x' and a column labelled 'y'
            that correspod to the locations of each station (in the same CRS
            as the other rasters). Required if calibrating against station
            measurements. Ignored if providing `t_raster_filepaths`.
        dates : str or datetime-like or list-like, optional
            Date or list of dates that correspond to each of the observed
            temperature raster provided in `t_raster_filepaths`. Ignored if
            `station_t_filepath` is provided.
        align_rasters : bool, default True
            Whether the rasters should be aligned before passing them as
            arguments of the InVEST urban cooling model. Since the model
            already aligns the LULC and reference evapotranspiration rasters,
            this argument is only useful to align the temperature rasters, and
            is therefore ignored if calibrating against station measurements.
        workspace_dir : str, optional
            Path to the folder where the model outputs will be written. If not
            provided, a temporary directory will be used.
        extra_ucm_args : dict-like, optional
            Other keyword arguments to be passed to the `execute` method of
            the urban cooling model.
        num_workers : int, optional
            Number of workers so that the simulations of each iteration can be
            executed at scale. Only useful if calibrating for multiple dates.
            If not provided, it will be set automatically depending on the
            number of dates and available number of processors in the CPU.
        """

        if workspace_dir is None:
            # TODO: how do we ensure that this is removed?
            workspace_dir = tempfile.mkdtemp()
            # TODO: log to warn that we are using a temporary directory
        # self.workspace_dir = workspace_dir
        # self.base_args.update()

        # evapotranspiration rasters for each date
        if isinstance(ref_et_raster_filepaths, str):
            ref_et_raster_filepaths = [ref_et_raster_filepaths]

        # get the raster metadata from lulc (used to predict air temperature
        # rasters)
        with rio.open(lulc_raster_filepath) as src:
            meta = src.meta.copy()
            data_mask = src.dataset_mask().astype(bool)

        # calibration approaches
        if t_raster_filepaths is not None:
            # calibrate against a map
            if isinstance(t_raster_filepaths, str):
                t_raster_filepaths = [t_raster_filepaths]

            if align_rasters:
                # a list is needed for the `_align_rasters` method
                if isinstance(ref_et_raster_filepaths, tuple):
                    ref_et_raster_filepaths = list(ref_et_raster_filepaths)
                if isinstance(t_raster_filepaths, tuple):
                    t_raster_filepaths = list(t_raster_filepaths)
                # align the rasters to the LULC raster and dump them to new
                # paths in the workspace directory
                dst_lulc_raster_filepath = path.join(workspace_dir, 'lulc.tif')
                dst_ref_et_raster_filepaths = [
                    path.join(workspace_dir, f'ref-et_{i}.tif')
                    for i in range(len(t_raster_filepaths))
                ]
                dst_t_raster_filepaths = [
                    path.join(workspace_dir, f't_{i}.tif')
                    for i in range(len(t_raster_filepaths))
                ]
                # the call below returns the same `dst_lulc_raster_filepath`
                # `dst_ref_et_raster_filepaths` and `dst_t_raster_filepaths`
                # passed as args
                (meta, data_mask, lulc_raster_filepath,
                 ref_et_raster_filepaths, t_raster_filepaths) = _align_rasters(
                     lulc_raster_filepath, ref_et_raster_filepaths,
                     t_raster_filepaths, dst_lulc_raster_filepath,
                     dst_ref_et_raster_filepaths, dst_t_raster_filepaths)

            # observed values array, Tref and UHImax
            if t_refs is None:
                if uhi_maxs is None:
                    obs_arrs, t_refs, uhi_maxs = _preprocess_t_rasters(
                        t_raster_filepaths)
                else:
                    obs_arrs, t_refs, _ = _preprocess_t_rasters(
                        t_raster_filepaths)
            else:
                if uhi_maxs is None:
                    obs_arrs, _, uhi_maxs = _preprocess_t_rasters(
                        t_raster_filepaths)
                else:
                    obs_arrs, _, __ = _preprocess_t_rasters(t_raster_filepaths)
            # need to replace nodata with `nan` so that `dropna` works below
            # the `_preprocess_t_rasters` method already uses `np.where` to
            # that end, however the `data_mask` used here might be different
            # (i.e., the intersection of the data regions of all rasters)
            obs_arr = np.concatenate([
                np.where(data_mask, _obs_arr, np.nan) for _obs_arr in obs_arrs
            ])

            # attributes to index the samples
            if isinstance(dates, str):
                dates = [dates]
            sample_name = 'pixel'
            # the sample index/keys here will select all the pixels of the
            # rasters, indexed by their flat-array position - this is rather
            # silly but this way the attributes work in the same way when
            # calibrating against observed temperature rasters or station
            # measurements
            # sample_keys = np.flatnonzero(data_mask)
            # sample_index = np.arange(data_mask.sum())
            sample_index = np.arange(data_mask.size)
            sample_keys = np.arange(data_mask.size)
        elif station_t_filepath is not None:
            station_location_df = pd.read_csv(station_locations_filepath,
                                              index_col=0)
            station_t_df = pd.read_csv(station_t_filepath,
                                       index_col=0)[station_location_df.index]
            station_t_df.index = pd.to_datetime(station_t_df.index)

            # observed values array, Tref and UHImax
            if t_refs is None:
                t_refs = station_t_df.min(axis=1)
            if uhi_maxs is None:
                uhi_maxs = station_t_df.max(axis=1) - t_refs
            obs_arr = station_t_df.values  # .flatten()

            # attributes to index the samples
            dates = station_t_df.index
            sample_name = 'station'
            sample_index = station_t_df.columns
            sample_keys = np.ravel_multi_index(
                transform.rowcol(meta['transform'], station_location_df['x'],
                                 station_location_df['y']),
                (meta['height'], meta['width']))
        else:
            # this is useful in this same method (see below)
            dates = None
            sample_name = None
            sample_index = None
            sample_keys = None
            obs_arr = None

        # create a dummy geojson with the bounding box extent for the area of
        # interest - this is completely ignored during the calibration
        aoi_vector_filepath = path.join(workspace_dir, 'dummy_aoi.geojson')
        with rio.open(lulc_raster_filepath) as src:
            # geom = geometry.box(*src.bounds)
            with fiona.open(
                    aoi_vector_filepath, 'w', driver='GeoJSON', crs=src.crs,
                    schema={
                        'geometry': 'Polygon',
                        'properties': {
                            'id': 'int'
                        }
                    }) as c:
                c.write({
                    'geometry': geometry.mapping(geometry.box(*src.bounds)),
                    'properties': {
                        'id': 1
                    },
                })

        # store the attributes to index the samples
        self.meta = meta
        self.data_mask = data_mask
        self.dates = dates
        self.sample_name = sample_name
        self.sample_index = sample_index
        self.sample_keys = sample_keys

        # store reference temperatures and UHI magnitudes as class attributes
        if not _is_sequence(t_refs):
            t_refs = [t_refs]
        if not _is_sequence(uhi_maxs):
            uhi_maxs = [uhi_maxs]
        self.t_refs = t_refs
        self.uhi_maxs = uhi_maxs

        # flat observation array to compute the calibration metric
        if obs_arr is not None:
            self.obs_arr = obs_arr.flatten()
            self.obs_mask = ~np.isnan(self.obs_arr)
            self.obs_arr = self.obs_arr[self.obs_mask]

        # model parameters: prepare the dict here so that all the paths/
        # parameters have been properly set above
        self.base_args = {
            'lulc_raster_path': lulc_raster_filepath,
            'biophysical_table_path': biophysical_table_filepath,
            'aoi_vector_path': aoi_vector_filepath,
            'cc_method': cc_method,
            'workspace_dir': workspace_dir,
        }
        # if model_params is None:
        #     model_params = DEFAULT_MODEL_PARAMS
        self.base_args.update(**settings.DEFAULT_UCM_PARAMS)

        if extra_ucm_args is None:
            extra_ucm_args = settings.DEFAULT_EXTRA_UCM_ARGS
        if 'do_valuation' not in extra_ucm_args:
            extra_ucm_args['do_valuation'] = settings.DEFAULT_EXTRA_UCM_ARGS[
                'do_valuation']
        self.base_args.update(**extra_ucm_args)
        # also store the paths to the evapotranspiration rasters
        self.ref_et_raster_filepaths = ref_et_raster_filepaths

        # number of workers to perform each calibration iteration at scale
        if num_workers is None:
            num_workers = min(len(self.ref_et_raster_filepaths),
                              os.cpu_count())
        self.num_workers = num_workers

    # properties to process the geospatial raster grid
    @property
    def grid_x(self):
        try:
            return self._grid_x
        except AttributeError:
            cols = np.arange(self.meta['width'])
            x, _ = transform.xy(self.meta['transform'], cols, cols)
            self._grid_x = x
            return self._grid_x

    @property
    def grid_y(self):
        try:
            return self._grid_y
        except AttributeError:
            rows = np.arange(self.meta['height'])
            _, y = transform.xy(self.meta['transform'], rows, rows)
            self._grid_y = y
            return self._grid_y

    # methods to predict temperatures
    def predict_t_arr(self, i, ucm_args=None):
        """
        Predict a temperature array for one of the calibration dates

        Parameters
        ----------
        i : int
            Positional index of the calibration date
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of
            the urban cooling model. The provided keys will override those set
            in the `base_args` attribute of this class (set up in the
            initialization method).

        Returns
        -------
        t_arr : np.ndarray
            Predicted temperature array aligned with the LULC raster for the
            selected date
        """

        args = self.base_args.copy()
        if ucm_args is not None:
            args.update(ucm_args)
        # TODO: support unaligned rasters?
        # if read_kws is None:
        #     read_kws = {}

        # note that this workspace_dir corresponds to this date only
        workspace_dir = path.join(self.base_args['workspace_dir'], str(i))
        args.update(
            workspace_dir=workspace_dir,
            ref_eto_raster_path=self.ref_et_raster_filepaths[i],
            # t_ref=Tref_da.sel(time=date).item(),
            # uhi_max=uhi_max_da.sel(time=date).item()
            t_ref=self.t_refs[i],
            uhi_max=self.uhi_maxs[i])
        ucm.execute(args)

        with rio.open(
                path.join(args['workspace_dir'], 'intermediate',
                          'T_air.tif')) as src:
            # return src.read(1, **read_kws)
            return src.read(1)

    def predict_t_arrs(self, ucm_args=None):
        """
        Predict the temperatures arrays for all the calibration dates.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of
            the urban cooling model. The provided keys will override those set
            in the `base_args` attribute of this class (set up in the
            initialization method).

        Returns
        -------
        t : list of np.ndarray
            Predicted temperature arrays for each date
        """
        # we could also iterate over `self.t_refs` or `self.uhi_maxs`
        pred_delayed = [
            dask.delayed(self.predict_t_arr)(i, ucm_args)
            for i in range(len(self.ref_et_raster_filepaths))
        ]

        return list(
            dask.compute(*pred_delayed, scheduler='processes',
                         num_workers=self.num_workers))

    def predict_t_da(self, ucm_args=None):
        """
        Predict a temperature data-array aligned with the LULC raster for all
        the calibration dates

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of
            the urban cooling model. The provided keys will override those set
            in the `base_args` attribute of this class (set up in the
            initialization method).

        Returns
        -------
        t_da : xr.DataArray
            Predicted temperature data array aligned with the LULC raster
        """

        t_arrs = self.predict_t_arrs(ucm_args=ucm_args)

        if self.dates is None:
            dates = np.arange(len(self.ref_et_raster_filepaths))
        else:
            dates = self.dates
        t_da = xr.DataArray(
            t_arrs, dims=('time', 'y', 'x'), coords={
                'time': dates,
                'y': self.grid_y,
                'x': self.grid_x
            }, name='T', attrs={'pyproj_srs': self.meta['crs'].to_proj4()})
        return t_da.groupby('time').apply(
            lambda x: x.where(self.data_mask, np.nan))

    def get_sample_comparison_df(self, ucm_args=None):
        """
        Compute a comparison data frame of the observed and predicted values
        for each sample (i.e., station measurement for a specific date).
        Requires that the object has been instantiated with either
        `t_raster_filepath` or `station_t_filepath`.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of
            the urban cooling model. The provided keys will override those set
            in the `base_args` attribute of this class (set up in the
            initialization method).

        Returns
        -------
        sample_comparison_df : pd.DataFrame
            Comparison data frame with columns for the sample date, station,
            observed and predicted values
        """

        tair_pred_df = pd.DataFrame(index=self.sample_index)

        t_da = self.predict_t_da(ucm_args=ucm_args)
        for date, date_da in t_da.groupby('time'):
            tair_pred_df[date] = date_da.values.flatten()[self.sample_keys]
        tair_pred_df = tair_pred_df.transpose()

        # comparison_df['err'] = comparison_df['pred'] - comparison_df['obs']
        # comparison_df['sq_err'] = comparison_df['err']**2
        sample_comparison_df = pd.DataFrame(
            {'pred': tair_pred_df.stack(dropna=False)[self.obs_mask]})
        sample_comparison_df.loc[sample_comparison_df.index,
                                 'obs'] = self.obs_arr
        return sample_comparison_df.reset_index().rename(columns={
            'level_0': 'date',
            'level_1': self.sample_name,
            0: 'obs',
            1: 'pred'
        })

    def get_model_perf_df(self, ucm_args=None, num_runs=None):
        """
        Compute comparing the performance of the calibrated model with
        randomly sampling temperature values from the
        :math:`[T_{ref}, T_{ref} + UHI_{max}]` range according to a uniform
        and normal distribution. Requires that the object has been
        instantiated with either `t_raster_filepath` or `station_t_filepath`.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of
            the urban cooling model. The provided keys will override those set
            in the `base_args` attribute of this class (set up in the
            initialization method).
        num_runs : int, optional
            Number of runs over which the results of randomly sampling (from
            both the uniform and normal distribution) will be averaged. If not
            provided, the value set in `settings.DEFAULT_MODEL_PERF_NUM_RUNS`
            will be used.

        Returns
        -------
        model_perf_df : pd.DataFrame
            Predicted temperature data array aligned with the LULC raster
        """

        comparison_df = self.get_sample_comparison_df(
            ucm_args=ucm_args).dropna()

        if num_runs is None:
            num_runs = settings.DEFAULT_MODEL_PERF_NUM_RUNS
        uniform_values = []
        normal_values = []
        for _ in range(num_runs):
            for date, date_df in comparison_df.groupby('date'):
                date_obs_ser = date_df['obs']
                T_min = date_obs_ser.min()
                T_max = date_obs_ser.max()
                num_samples = len(date_obs_ser)
                uniform_values.append(
                    np.random.uniform(T_min, T_max, size=num_samples))
                normal_values.append(
                    np.random.normal(loc=date_df['obs'].mean(),
                                     scale=date_df['obs'].std(),
                                     size=num_samples))
        uniform_values = np.concatenate(uniform_values)
        normal_values = np.concatenate(normal_values)

        model_perf_df = pd.DataFrame(columns=METRIC_COLUMNS)

        # Uniform/normal
        obs_values = pd.concat([comparison_df['obs'] for _ in range(num_runs)])
        model_perf_df.loc['uniform'] = _compute_model_perf(
            obs_values, uniform_values)
        model_perf_df.loc['normal'] = _compute_model_perf(
            obs_values, normal_values)

        # InVEST urban cooling model
        model_perf_df.loc['invest_ucm'] = _compute_model_perf(
            comparison_df['obs'], comparison_df['pred'])

        return model_perf_df


class UCMCalibrator(simanneal.Annealer):
    def __init__(self, lulc_raster_filepath, biophysical_table_filepath,
                 cc_method, ref_et_raster_filepaths, t_refs=None,
                 uhi_maxs=None, t_raster_filepaths=None,
                 station_t_filepath=None, station_locations_filepath=None,
                 dates=None, align_rasters=True, workspace_dir=None,
                 initial_solution=None, extra_ucm_args=None, metric=None,
                 stepsize=None, exclude_zero_kernel_dist=True,
                 num_workers=None, num_steps=None, num_update_logs=None):
        """
        Utility to calibrate the urban cooling model

        Parameters
        ----------
        lulc_raster_filepath : str
            Path to the raster of land use/land cover (LULC) file
        biophysical_table_filepath : str
            Path to the biophysical table CSV file
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
        dates : str or datetime-like or list-like, optional
            Date or list of dates that correspond to each of the observed
            temperature raster provided in `t_raster_filepaths`. Ignored if
            `station_t_filepath` is provided.
        align_rasters : bool, default True
            Whether the rasters should be aligned before passing them as
            arguments of the InVEST urban cooling model. Since the model
            already aligns the LULC and reference evapotranspiration rasters,
            this argument is only useful to align the temperature rasters, and
            is therefore ignored if calibrating against station measurements.
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
        exclude_zero_kernel_dist : bool, default True
            Whether the calibration should consider parameters that lead to
            decay functions with a kernel distance of zero pixels (i.e.,
            `t_air_average_radius` or `green_area_cooling_distance` lower than
            half the LULC pixel resolution).
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
            lulc_raster_filepath, biophysical_table_filepath, cc_method,
            ref_et_raster_filepaths, t_refs=t_refs, uhi_maxs=uhi_maxs,
            t_raster_filepaths=t_raster_filepaths,
            station_t_filepath=station_t_filepath,
            station_locations_filepath=station_locations_filepath, dates=dates,
            align_rasters=align_rasters, workspace_dir=workspace_dir,
            extra_ucm_args=extra_ucm_args, num_workers=num_workers)

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
            initial_solution = list(settings.DEFAULT_UCM_PARAMS.values())
        # init the parent `Annealer` instance with the initial solution
        super(UCMCalibrator, self).__init__(initial_solution)

        # whether we ensure that kernel decay distances are of at least one
        # pixel
        if exclude_zero_kernel_dist:
            with rio.open(
                    self.ucm_wrapper.base_args['lulc_raster_path']) as src:
                # the chained `np.min` and `np.abs` corresponds to the way
                # that the urban cooling model sets the `cell_size` variable
                # which is in turn used in the denominator when obtaining
                # kernel distances
                self.min_kernel_dist = 0.5 * np.min(np.abs(
                    src.res)) + settings.MIN_KERNEL_DIST_EPS
        self.exclude_zero_kernel_dist = exclude_zero_kernel_dist

        # nicer parameters for the urban cooling model solution space
        if num_steps is None:
            num_steps = settings.DEFAULT_NUM_STEPS
        self.steps = num_steps
        if num_update_logs is None:
            num_update_logs = settings.DEFAULT_NUM_UPDATE_LOGS
        self.updates = num_update_logs

    # property to get the model parameters according to the calibration state
    @property
    def _ucm_params_dict(self):
        return dict(t_air_average_radius=self.state[0],
                    green_area_cooling_distance=self.state[1],
                    cc_weight_shade=self.state[2],
                    cc_weight_albedo=self.state[3],
                    cc_weight_eti=self.state[4])

    # methods required so that the `Annealer` class works for our purpose
    def move(self):
        state_neighbour = []
        for param in self.state:
            state_neighbour.append(
                param * (1 + rn.uniform(-self.stepsize, self.stepsize)))
        # ensure that kernel decay distances are of at least one pixel
        if self.exclude_zero_kernel_dist:
            for k in range(2):
                if state_neighbour[k] < self.min_kernel_dist:
                    state_neighbour[k] = self.min_kernel_dist
                # alternatively:
                # state_neighbour[k] = np.max(state_neighbour[k],
                #                             self.min_kernel_dist)
        # rescale so that the three weights add up to one
        weight_sum = sum(state_neighbour[2:])
        for k in range(2, 5):
            state_neighbour[k] /= weight_sum

        # update the state
        self.state = state_neighbour

    def energy(self):
        ucm_args = self._ucm_params_dict.copy()
        pred_arr = np.hstack(self.ucm_wrapper.predict_t_arrs(
            ucm_args=ucm_args)).flatten()[self.ucm_wrapper.sample_keys]

        return self.compute_metric(self.ucm_wrapper.obs_arr,
                                   pred_arr[self.ucm_wrapper.obs_mask])

    def calibrate(self, initial_solution=None, num_steps=None,
                  num_update_logs=None):
        """
        Run a simulated annealing procedure to get the arguments of the InVEST
        urban cooling model that minimize the performance metric

        Parameters
        ----------
        initial_solution : list-like, optional
            Sequence with the parameter values used as initial solution, of
            the form (t_air_average_radius, green_area_cooling_distance,
            cc_weight_shade, cc_weight_albedo, cc_weight_eti). If not provided,
            the default values of the urban cooling model will be used.
        num_steps : int, optional.
            Number of iterations of the simulated annealing procedure. If not
            provided, the value set in `settings.DEFAULT_NUM_STEPS` will be
            used.
        num_update_logs : int, default 100
            Number of updates that will be logged. If `num_steps` is equal to
            `num_update_logs`, each iteration will be logged. If not provided,
            the value set in `settings.DEFAULT_UPDATE_LOGS` will be used.

        Returns
        -------
        (state, metric) : the best state, i.e., combination of arguments of
            the form (t_air_average_radius, green_area_cooling_distance,
            cc_weight_shade, cc_weight_albedo, cc_weight_eti) and the
            corresponding metric
        """

        # Override the values set in the init method. Note that the attribute
        # names are defined in the `Annealer` class
        if initial_solution is not None:
            self.state = initial_solution
        if num_steps is not None:
            self.steps = num_steps
        if num_update_logs is not None:
            self.updates = num_update_logs

        return self.anneal()

    # shortcuts to useful `UCMWrapper` methods
    # TODO: dry `ucm_args` with a decorator?
    def predict_t_arrs(self, ucm_args=None):
        """        
        Predict the temperatures arrays for all the calibration dates.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of
            the urban cooling model. The provided keys will override those set
            in the `base_args` attribute of this class (set up in the
            initialization method).

        Returns
        -------
        t : np.ndarray
            Predicted temperature arrays for each date
        """

        if ucm_args is None:
            ucm_args = self._ucm_params_dict.copy()

        return self.ucm_wrapper.predict_t_arrs(ucm_args=ucm_args)

    def predict_t_da(self, ucm_args=None):
        """
        Predict a temperature data-array aligned with the LULC raster for all
        the calibration dates

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of
            the urban cooling model. The provided keys will override those set
            in the current solution found by the calibrator, i.e., the `state`
            attribute.

        Returns
        -------
        t_da : xr.DataArray
            Predicted temperature data array aligned with the LULC raster
        """

        if ucm_args is None:
            ucm_args = self._ucm_params_dict.copy()

        return self.ucm_wrapper.predict_t_da(ucm_args=ucm_args)

    def get_sample_comparison_df(self, ucm_args=None):
        """
        Compute a comparison data frame of the observed and predicted values
        for each sample (i.e., station measurement for a specific date)

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of
            the urban cooling model. The provided keys will override those set
            in the current solution found by the calibrator, i.e., the `state`
            attribute.

        Returns
        -------
        sample_comparison_df : pd.DataFrame
            Comparison data frame with columns for the sample date, station,
            observed and predicted values
        """

        if ucm_args is None:
            ucm_args = self._ucm_params_dict.copy()

        return self.ucm_wrapper.get_sample_comparison_df(ucm_args=ucm_args)

    def get_model_perf_df(self, ucm_args=None, num_runs=None):
        """
        Compute comparing the performance of the calibrated model with
        randomly sampling temperature values from the
        :math:`[T_{ref}, T_{ref} + UHI_{max}]` range according to a uniform
        and normal distribution

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of
            the urban cooling model. The provided keys will override those set
            in the current solution found by the calibrator, i.e., the `state`
            attribute.
        num_runs : int, optional
            Number of runs over which the results of randomly sampling (from
            both the uniform and normal distribution) will be averaged. If not
            provided, the value set in `settings.DEFAULT_MODEL_PERF_NUM_RUNS`
            will be used.

        Returns
        -------
        model_perf_df : pd.DataFrame
            Predicted temperature data array aligned with the LULC raster
        """

        if ucm_args is None:
            ucm_args = self._ucm_params_dict.copy()

        return self.ucm_wrapper.get_model_perf_df(ucm_args=ucm_args,
                                                  num_runs=num_runs)
