"""InVEST Urban Cooling Model calibration module."""
import os
import tempfile
from os import path

import fiona
import numpy as np
import numpy.random as rn
import pandas as pd
import rasterio as rio
import rioxarray as rxr
import simanneal
import xarray as xr
from natcap.invest import urban_cooling_model as ucm
from rasterio import transform, warp
from rasterio.warp import Resampling
from scipy import stats
from shapely import geometry
from sklearn import metrics

from . import settings

__version__ = "0.6.0"


# utils
def _is_sequence(arg):
    # # Based on steveha's answer in stackoverflow https://bit.ly/3dpnf0m
    # return (not hasattr(arg, "strip") and hasattr(arg, "__getitem__")
    #         or hasattr(arg, "__iter__"))
    return hasattr(arg, "__getitem__") or hasattr(arg, "__iter__")


def _date_workspace_dir(base_workspace_dir, i):
    # very simple function to avoid duplicating the logic of generating a date-specific
    # workspace dir
    return path.join(base_workspace_dir, str(i))


def _preprocess_t_rasters(t_raster_filepaths, dst_meta, *, resampling=None):
    # get dst raster info (constant) outside the loop
    dst_shape = (dst_meta["height"], dst_meta["width"])
    dst_transform = dst_meta["transform"]
    dst_crs = dst_meta["crs"]
    if resampling is None:
        resampling = Resampling.bilinear

    # init empty arrays
    obs_arrs = []
    t_refs = []
    uhi_maxs = []
    for t_raster_filepath in t_raster_filepaths:
        with rio.open(t_raster_filepath) as src:
            dst_arr = np.full(dst_shape, src.nodata, dtype=src.dtypes[0])
            src_mask = src.dataset_mask()
            dst_mask = np.zeros(dst_shape, dtype=src_mask.dtype)
            # by default, when reprojecting with a source rasterio dataset and a
            # destination array, the source nodata and destination nodata are set to
            # src.nodata, so we do not need to handle it
            warp.reproject(
                source=src.read(1),
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=src.nodata,
                resampling=resampling,
            )

            warp.reproject(
                source=src_mask,
                destination=dst_mask,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )

        # t_arr = src.read(1)
        # use `np.nan` for nodata values to ensure that we get the right min/max values
        # with `np.nanmin`/`np.nanmax`
        t_arr = np.where(dst_mask, dst_arr, np.nan)
        obs_arrs.append(t_arr)
        t_min = np.nanmin(t_arr)
        t_refs.append(t_min)
        uhi_maxs.append(np.nanmax(t_arr) - t_min)

    return obs_arrs, t_refs, uhi_maxs


def _r2_score(obs, pred):
    slope, intercept, r_value, p_value, std_err = stats.linregress(obs, pred)
    return r_value * r_value


def _inverted_r2_score(obs, pred):
    # since we need to maximize (instead of minimize) the r2, the simulated annealing
    # will actually minimize 1 - R^2
    return 1 - _r2_score(obs, pred)


METRIC_COLUMNS = ["R^2", "MAE", "RMSE"]


def _compute_model_perf(obs, pred):
    return [
        _r2_score(obs, pred),
        metrics.mean_absolute_error(obs, pred),
        metrics.mean_squared_error(obs, pred, squared=False),
    ]


# classes
class UCMWrapper:
    """
    Pythonic and open source interface to the InVEST urban cooling model.

    A set of additional utility methods serve to compute temperature maps and data
    frames.
    """

    def __init__(
        self,
        lulc_raster_filepath,
        biophysical_table_filepath,
        cc_method,
        ref_et_raster_filepaths,
        *,
        aoi_vector_filepath=None,
        t_refs=None,
        uhi_maxs=None,
        t_raster_filepaths=None,
        station_t_filepath=None,
        station_locations_filepath=None,
        dates=None,
        workspace_dir=None,
        extra_ucm_args=None,
    ):
        """
        Initialize a Pythonic Urban Cooling Model wrapper.

        Parameters
        ----------
        lulc_raster_filepath : str
            Path to the raster of land use/land cover (LULC) file.
        biophysical_table_filepath : str
            Path to the biophysical table CSV file.
        cc_method : str
            Cooling capacity calculation method. Can be either 'factors' or 'intensity'.
        ref_et_raster_filepaths : str or list-like
            Path to the reference evapotranspiration raster, or sequence of strings with
            a path to the reference evapotranspiration raster.
        aoi_vector_filepath : str, optional
            Path to the area of interest vector. If not provided, the bounds of the LULC
            raster will be used.
        t_refs : numeric or list-like, optional
            Reference air temperature. If not provided, it will be set as the minimum
            observed temperature (raster or station measurements, for each respective
            date if calibrating for multiple dates).
        uhi_maxs : numeric or list-like, optional
            Magnitude of the UHI effect. If not provided, it will be set as the
            difference between the maximum and minimum observed temperature (raster or
            station measurements, for each respective date if calibrating for multiple
            dates).
        t_raster_filepaths : str or list-like, optional
            Path to the observed temperature raster, or sequence of strings with a path
            to the observed temperature rasters. Required if calibrating against
            temperature map(s).
        station_t_filepath : str, optional
            Path to a table of air temperature measurements where each column
            corresponds to a monitoring station and each row to a datetime. Required if
            calibrating against station measurements. Ignored if providing
            `t_raster_filepaths`.
        station_locations_filepath : str, optional
            Path to a table with the locations of each monitoring station, where the
            first column features the station labels (that match the columns of the
            table of air temperature measurements), and there are (at least) a column
            labelled 'x' and a column labelled 'y' that correspod to the locations of
            each station (in the same CRS as the other rasters). Required if calibrating
            against station measurements. Ignored if providing `t_raster_filepaths`.
        dates : str or datetime-like or list-like, optional
            Date or list of dates that correspond to each of the observed temperature
            raster provided in `t_raster_filepaths`. Ignored if `station_t_filepath` is
            provided.
        workspace_dir : str, optional
            Path to the folder where the model outputs will be written. If not provided,
            a temporary directory will be used.
        Extra_ucm_args : dict-like, optional
            Other keyword arguments to be passed to the `execute` method of the urban
            cooling model.
        """
        # 1. prepare base args
        # 1.1. workspace dir
        if workspace_dir is None:
            # TODO: how do we ensure that this is removed?
            workspace_dir = tempfile.mkdtemp()
            # TODO: log to warn that we are using a temporary directory
        # self.workspace_dir = workspace_dir
        # self.base_args.update()

        # 1.2. area of interest vector
        # create a dummy geojson with the bounding box extent for the area of interest -
        # this is completely ignored during the calibration
        if aoi_vector_filepath is None:
            aoi_vector_filepath = path.join(workspace_dir, "dummy_aoi.geojson")
            with rio.open(lulc_raster_filepath) as src:
                # geom = geometry.box(*src.bounds)
                with fiona.open(
                    aoi_vector_filepath,
                    "w",
                    driver="GeoJSON",
                    crs=src.crs.to_string(),
                    schema={"geometry": "Polygon", "properties": {"id": "int"}},
                ) as c:
                    c.write(
                        {
                            "geometry": geometry.mapping(geometry.box(*src.bounds)),
                            "properties": {"id": 1},
                        }
                    )

        # 1.3. set base args dict as instance attribute
        # model parameters: prepare the dict here so that all the paths/ parameters have
        # been properly set above
        self.base_args = {
            "lulc_raster_path": lulc_raster_filepath,
            "biophysical_table_path": biophysical_table_filepath,
            "aoi_vector_path": aoi_vector_filepath,
            "cc_method": cc_method,
            "workspace_dir": workspace_dir,
        }
        # if model_params is None:
        #     model_params = DEFAULT_MODEL_PARAMS
        # set the basic model params used in all methods
        self.base_args.update(**settings.DEFAULT_UCM_PARAMS)
        # for the "factors" cooling capacity method,
        if self.base_args["cc_method"] == "factors":
            self.base_args.update(**settings.DEFAULT_UCM_FACTORS_PARAMS)

        if extra_ucm_args is None:
            extra_ucm_args = settings.DEFAULT_EXTRA_UCM_ARGS
        # if the user provides custom `extra_ucm_args`, check that the required args are
        # set
        for extra_ucm_arg in settings.DEFAULT_EXTRA_UCM_ARGS:
            if extra_ucm_arg not in extra_ucm_args:
                extra_ucm_args[extra_ucm_arg] = settings.DEFAULT_EXTRA_UCM_ARGS[
                    extra_ucm_arg
                ]
        self.base_args.update(**extra_ucm_args)

        # 2. evapotranspiration rasters
        # evapotranspiration rasters for each date
        if isinstance(ref_et_raster_filepaths, str):
            ref_et_raster_filepaths = [ref_et_raster_filepaths]

        # also store the paths to the evapotranspiration rasters
        self.ref_et_raster_filepaths = ref_et_raster_filepaths

        # 3. raster metadata
        # get the raster metadata from lulc (used to predict air temperature rasters)
        # with rio.open(lulc_raster_filepath) as src:
        #     meta = src.meta.copy()
        #     data_mask = src.dataset_mask().astype(bool)
        # run the model once in a temporary directory so that we get the raster metadata
        # from the run's outputs. This way we do not need to duplicate their raster
        # alignment procedure in our code.
        args = self.base_args.copy()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args.update(
                workspace_dir=tmp_dir,
                ref_eto_raster_path=ref_et_raster_filepaths[0],
                t_ref=0,
                uhi_max=1,
            )
            ucm.execute(args)

            with rio.open(
                path.join(args["workspace_dir"], "intermediate", "T_air.tif")
            ) as src:
                # return src.read(1, **read_kws)
                meta = src.meta.copy()
                data_mask = src.dataset_mask().astype(bool)

        # calibration approaches
        if t_raster_filepaths is not None:
            # calibrate against a map
            if isinstance(t_raster_filepaths, str):
                t_raster_filepaths = [t_raster_filepaths]

            # observed values array, Tref and UHImax
            if t_refs is None:
                if uhi_maxs is None:
                    obs_arrs, t_refs, uhi_maxs = _preprocess_t_rasters(
                        t_raster_filepaths, meta
                    )
                else:
                    obs_arrs, t_refs, _ = _preprocess_t_rasters(
                        t_raster_filepaths, meta
                    )
            else:
                if uhi_maxs is None:
                    obs_arrs, _, uhi_maxs = _preprocess_t_rasters(
                        t_raster_filepaths, meta
                    )
                else:
                    obs_arrs, _, __ = _preprocess_t_rasters(t_raster_filepaths, meta)
            # need to replace nodata with `nan` so that `dropna` works below the
            # `_preprocess_t_rasters` method already uses `np.where` to that end,
            # however the `data_mask` used here might be different (i.e., the
            # intersection of the data regions of all rasters)
            obs_arr = np.concatenate(
                [np.where(data_mask, _obs_arr, np.nan) for _obs_arr in obs_arrs]
            )

            # attributes to index the samples
            sample_name = "pixel"
            # the sample index/keys here will select all the pixels of the rasters,
            # indexed by their flat-array position - this is rather silly but this way
            # the attributes work in the same way when calibrating against observed
            # temperature rasters or station measurements
            # sample_keys = np.flatnonzero(data_mask)
            # sample_index = np.arange(data_mask.sum())
            sample_index = np.arange(data_mask.size)
            sample_keys = np.arange(data_mask.size)
        elif station_t_filepath is not None:
            station_location_df = pd.read_csv(station_locations_filepath, index_col=0)
            station_t_df = pd.read_csv(station_t_filepath, index_col=0)[
                station_location_df.index
            ]
            station_t_df.index = pd.to_datetime(station_t_df.index)

            # observed values array, Tref and UHImax
            if t_refs is None:
                t_refs = station_t_df.min(axis=1)
            if uhi_maxs is None:
                uhi_maxs = station_t_df.max(axis=1) - t_refs
            obs_arr = station_t_df.values  # .flatten()

            # attributes to index the samples
            dates = station_t_df.index
            sample_name = "station"
            sample_index = station_t_df.columns
            sample_keys = np.ravel_multi_index(
                transform.rowcol(
                    meta["transform"],
                    station_location_df["x"],
                    station_location_df["y"],
                ),
                (meta["height"], meta["width"]),
            )
        else:
            # this is useful in this same method (see below)
            sample_name = None
            sample_index = None
            sample_keys = None
            obs_arr = None

        # process the dates attribute
        if isinstance(dates, str):
            # if at this point dates is a string, it means that it has been passed as a
            # string in the init argument. We have to make it a list so that we can
            # iterate it properly
            dates = [dates]
        elif dates is None:
            # if at this point dates is None, let us just make it an integer list
            dates = np.arange(len(self.ref_et_raster_filepaths))

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

    # methods to predict temperatures
    def predict_t_arr(self, i, *, ucm_args=None):
        """
        Predict a temperature array for one of the calibration dates.

        Parameters
        ----------
        i : int
            Positional index of the calibration date.
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of the urban
            cooling model. The provided keys will override those set in the `base_args`
            attribute of this class (set up in the initialization method).

        Returns
        -------
        t_arr : np.ndarray
            Predicted temperature array aligned with the LULC raster for the selected
            date.
        """
        args = self.base_args.copy()
        if ucm_args is not None:
            args.update(ucm_args)
        # TODO: support unaligned rasters?
        # if read_kws is None:
        #     read_kws = {}

        # if no specific workspace dir is provided in `ucm_args`, a dedicated
        # workspace_dir for this date only is used
        base_workspace_dir = self.base_args["workspace_dir"]
        if args["workspace_dir"] == base_workspace_dir:
            args.update(workspace_dir=_date_workspace_dir(base_workspace_dir, i))
        # update the rest of date-specific args
        args.update(
            ref_eto_raster_path=self.ref_et_raster_filepaths[i],
            # t_ref=Tref_da.sel(time=date).item(),
            # uhi_max=uhi_max_da.sel(time=date).item()
            t_ref=self.t_refs[i],
            uhi_max=self.uhi_maxs[i],
        )
        ucm.execute(args)

        with rio.open(
            path.join(args["workspace_dir"], "intermediate", "T_air.tif")
        ) as src:
            # return src.read(1, **read_kws)
            return src.read(1)

    def predict_t_da(self, *, ucm_args=None):
        """
        Predict a temperature data-array.

        The array is aligned with the LULC raster for all the calibration dates.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of the urban
            cooling model. The provided keys will override those set in the `base_args`
            attribute of this class (set up in the initialization method).

        Returns
        -------
        t_da : xr.DataArray
            Predicted temperature data array aligned with the LULC raster.
        """
        if ucm_args is None:
            ucm_args = {}
        workspace_dirs = []
        base_workspace_dir = self.base_args["workspace_dir"]
        for i, _ in enumerate(self.dates):
            workspace_dir = _date_workspace_dir(base_workspace_dir, i)
            workspace_dirs.append(workspace_dir)
            ucm_args["workspace_dir"] = workspace_dir
            _ = self.predict_t_arr(i, ucm_args=ucm_args)

        t_da = xr.concat(
            [
                rxr.open_rasterio(filepath)
                for filepath in [
                    path.join(workspace_dir, "intermediate", "T_air.tif")
                    for workspace_dir in workspace_dirs
                ]
            ],
            dim="time",
        )
        return t_da.where(t_da != t_da.attrs["_FillValue"])

    def get_sample_comparison_df(self, *, ucm_args=None):
        """
        Compute a comparison data frame of the observed and predicted values.

        Each row corresponds to a sample (i.e., station measurement for a specific
        date). Requires that the object has been instantiated with either
        `t_raster_filepath` or `station_t_filepath`.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of the urban
            cooling model. The provided keys will override those set in the `base_args`
            attribute of this class (set up in the initialization method).

        Returns
        -------
        sample_comparison_df : pd.DataFrame
            Comparison data frame with columns for the sample date, station, observed
            and predicted values.
        """
        tair_pred_df = pd.DataFrame(index=self.sample_index)

        t_da = self.predict_t_da(ucm_args=ucm_args)
        for date, date_da in t_da.groupby("time"):
            tair_pred_df[date] = date_da.values.flatten()[self.sample_keys]
        tair_pred_df = tair_pred_df.transpose()

        # comparison_df['err'] = comparison_df['pred'] - comparison_df['obs']
        # comparison_df['sq_err'] = comparison_df['err']**2
        sample_comparison_df = pd.DataFrame(
            {"pred": tair_pred_df.stack(dropna=False)[self.obs_mask]}
        )
        sample_comparison_df.loc[sample_comparison_df.index, "obs"] = self.obs_arr
        return sample_comparison_df.reset_index().rename(
            columns={
                "level_0": "date",
                "level_1": self.sample_name,
                0: "obs",
                1: "pred",
            }
        )

    def get_model_perf_df(self, *, ucm_args=None, compare_random=None, num_runs=None):
        """
        Compute a model performance data frame.

        Compare the performance of the calibrated model with randomly sampling
        temperature values from the :math:`[T_{ref}, T_{ref} + UHI_{max}]` range
        according to a uniform and normal distribution. Requires that the object has
        been instantiated with either `t_raster_filepath` or `station_t_filepath`.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of the urban
            cooling model. The provided keys will override those set in the `base_args`
            attribute of this class (set up in the initialization method).
        compare_random : bool, optional
            Whether the performance of the urban cooling model should be compared to
            randomly sampling (from the uniform and normal distribution). If not
            provided, the value set in `settings.DEFAULT_MODEL_PERF_COMPARE_RANDOM` will
            be used.
        num_runs : int, optional
            Number of runs over which the results of randomly sampling (from both the
            uniform and normal distribution) will be averaged. If not provided, the
            value set in `settings.DEFAULT_MODEL_PERF_NUM_RUNS` will be used. Ignored if
            `compare_random` is False.

        Returns
        -------
        model_perf_df : pd.DataFrame
            Predicted temperature data array aligned with the LULC raster.
        """
        comparison_df = self.get_sample_comparison_df(ucm_args=ucm_args).dropna()

        model_perf_df = pd.DataFrame(columns=METRIC_COLUMNS)
        # InVEST urban cooling model
        model_perf_df.loc["invest_ucm"] = _compute_model_perf(
            comparison_df["obs"], comparison_df["pred"]
        )

        if compare_random is None:
            compare_random = settings.DEFAULT_MODEL_PERF_COMPARE_RANDOM

        if not compare_random:
            return model_perf_df

        if num_runs is None:
            num_runs = settings.DEFAULT_MODEL_PERF_NUM_RUNS
        uniform_values = []
        normal_values = []
        for _ in range(num_runs):
            for _date, date_df in comparison_df.groupby("date"):
                date_obs_ser = date_df["obs"]
                T_min = date_obs_ser.min()
                T_max = date_obs_ser.max()
                num_samples = len(date_obs_ser)
                uniform_values.append(np.random.uniform(T_min, T_max, size=num_samples))
                normal_values.append(
                    np.random.normal(
                        loc=date_df["obs"].mean(),
                        scale=date_df["obs"].std(),
                        size=num_samples,
                    )
                )
        uniform_values = np.concatenate(uniform_values)
        normal_values = np.concatenate(normal_values)

        # Uniform/normal
        obs_values = pd.concat([comparison_df["obs"] for _ in range(num_runs)])
        model_perf_df.loc["uniform"] = _compute_model_perf(obs_values, uniform_values)
        model_perf_df.loc["normal"] = _compute_model_perf(obs_values, normal_values)

        return model_perf_df


class UCMCalibrator(simanneal.Annealer):
    """Urban cooling model calibrator."""

    def __init__(
        self,
        lulc_raster_filepath,
        biophysical_table_filepath,
        cc_method,
        ref_et_raster_filepaths,
        *,
        aoi_vector_filepath=None,
        t_refs=None,
        uhi_maxs=None,
        t_raster_filepaths=None,
        station_t_filepath=None,
        station_locations_filepath=None,
        dates=None,
        workspace_dir=None,
        initial_solution=None,
        extra_ucm_args=None,
        metric=None,
        stepsize=None,
        exclude_zero_kernel_dist=None,
        num_steps=None,
        num_update_logs=None,
    ):
        """
        Initialize the urban cooling model calibrator.

        Parameters
        ----------
        lulc_raster_filepath : str
            Path to the raster of land use/land cover (LULC) file.
        biophysical_table_filepath : str
            Path to the biophysical table CSV file.
        cc_method : str
            Cooling capacity calculation method. Can be either 'factors' or 'intensity'.
        ref_et_raster_filepaths : str or list-like
            Path to the reference evapotranspiration raster, or sequence of strings with
            a path to the reference evapotranspiration raster.
        aoi_vector_filepath : str, optional
            Path to the area of interest vector. If not provided, the bounds of the LULC
            raster will be used.
        t_refs : numeric or list-like, optional
            Reference air temperature. If not provided, it will be set as the minimum
            observed temperature (raster or station measurements, for each respective
            date if calibrating for multiple dates).
        uhi_maxs : numeric or list-like, optional
            Magnitude of the UHI effect. If not provided, it will be set as the
            difference between the maximum and minimum observed temperature (raster or
            station measurements, for each respective date if calibrating for multiple
            dates).
        t_raster_filepaths : str or list-like, optional
            Path to the observed temperature raster, or sequence of strings with a path
            to the observed temperature rasters. Required if calibrating against
            temperature map(s).
        station_t_filepath : str, optional
            Path to a table of air temperature measurements where each column
            corresponds to a monitoring station and each row to a datetime.  Required if
            calibrating against station measurements.
        station_locations_filepath : str, optional
            Path to a table with the locations of each monitoring station, where the
            first column features the station labels (that match the columns of the
            table of air temperature measurements), and there are (at least) a column
            labelled 'x' and a column labelled 'y' that correspod to the locations of
            each station (in the same CRS as the other rasters). Required if calibrating
            against station measurements.
        dates : str or datetime-like or list-like, optional
            Date or list of dates that correspond to each of the observed temperature
            raster provided in `t_raster_filepaths`. Ignored if `station_t_filepath` is
            provided.
        workspace_dir : str, optional
            Path to the folder where the model outputs will be written. If not provided,
            a temporary directory will be used.
        initial_solution : list-like, optional
            Sequence with the parameter values used as initial solution, of the form
            (t_air_average_radius, green_area_cooling_distance, cc_weight_shade,
            cc_weight_albedo, cc_weight_eti). If not provided, the default values of the
            urban cooling model will be used.
        extra_ucm_args : dict-like, optional
            Other keyword arguments to be passed to the `execute` method of the urban
            cooling model.
        metric : {'R2', 'MAE', 'RMSE'}, optional
            Target metric to optimize in the calibration. Can be either 'R2' for the R
            squared (which will be maximized), 'MAE' for the mean absolute error (which
            will be minimized) or 'RMSE' for the (root) mean squared error (which will
            be minimized). If not provided, the value set in `settings.DEFAULT_METRIC`
            will be used.
        stepsize : numeric, optional
            Step size in terms of the fraction of each parameter when looking to select
            a neighbor solution for the following iteration. The neighbor will be
            randomly drawn from an uniform distribution in the [param - stepsize *
            param, param + stepsize * param] range. For example, with a step size of 0.3
            and a 't_air_average_radius' of 500 at a given iteration, the solution for
            the next iteration will be uniformly sampled from the [350, 650] range. If
            not provided, it will be taken from `settings.DEFAULT_STEPSIZE`.
        exclude_zero_kernel_dist : bool, optional.
            Whether the calibration should consider parameters that lead to decay
            functions with a kernel distance of zero pixels (i.e.,
            `t_air_average_radius` or `green_area_cooling_distance` lower than half the
            LULC pixel resolution). If not provided, the value set in
            `settings.DEFAULT_EXCLUDE_ZERO_KERNEL_DIST` will be used.
        num_steps : int, optional.
            Number of iterations of the simulated annealing procedure. If not provided,
            the value set in `settings.DEFAULT_NUM_STEPS` will be used.
        num_update_logs : int, default 100
            Number of updates that will be logged. If `num_steps` is equal to
            `num_update_logs`, each iteration will be logged. If not provided, the value
            set in `settings.DEFAULT_UPDATE_LOGS` will be used.
        """
        # init the model wrapper
        self.ucm_wrapper = UCMWrapper(
            lulc_raster_filepath,
            biophysical_table_filepath,
            cc_method,
            ref_et_raster_filepaths,
            aoi_vector_filepath=aoi_vector_filepath,
            t_refs=t_refs,
            uhi_maxs=uhi_maxs,
            t_raster_filepaths=t_raster_filepaths,
            station_t_filepath=station_t_filepath,
            station_locations_filepath=station_locations_filepath,
            dates=dates,
            workspace_dir=workspace_dir,
            extra_ucm_args=extra_ucm_args,
        )

        # metric
        if metric is None:
            metric = settings.DEFAULT_METRIC
        if metric == "R2":
            # since we need to maximize (instead of minimize) the r2, the simulated
            # annealing will actually minimize 1 - R^2
            self.compute_metric = _inverted_r2_score
        elif metric == "MAE":
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
            # for the factors cooling capacity method, also add its parameters to the
            # initial solution
            if self.ucm_wrapper.base_args["cc_method"] == "factors":
                initial_solution += list(settings.DEFAULT_UCM_FACTORS_PARAMS.values())
        # init the parent `Annealer` instance with the initial solution
        super().__init__(initial_solution)

        # whether we ensure that kernel decay distances are of at least one pixel
        if exclude_zero_kernel_dist is None:
            exclude_zero_kernel_dist = settings.DEFAULT_EXCLUDE_ZERO_KERNEL_DIST
        if exclude_zero_kernel_dist:
            with rio.open(self.ucm_wrapper.base_args["lulc_raster_path"]) as src:
                # the chained `np.min` and `np.abs` corresponds to the way that the
                # urban cooling model sets the `cell_size` variable which is in turn
                # used in the denominator when obtaining kernel distances
                self.min_kernel_dist = (
                    0.5 * np.min(np.abs(src.res)) + settings.MIN_KERNEL_DIST_EPS
                )
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
        ucm_params_dict = {
            "t_air_average_radius": self.state[0],
            "green_area_cooling_distance": self.state[1],
        }
        if self.ucm_wrapper.base_args["cc_method"] == "factors":
            ucm_params_dict.update(
                cc_weight_shade=self.state[2],
                cc_weight_albedo=self.state[3],
                cc_weight_eti=self.state[4],
            )
        return ucm_params_dict

    # methods required so that the `Annealer` class works for our purpose
    def move(self):
        """Change the model parameters to explore the best performance."""
        state_neighbour = []
        for param in self.state:
            state_neighbour.append(
                param * (1 + rn.uniform(-self.stepsize, self.stepsize))
            )
        # ensure that kernel decay distances are of at least one pixel
        if self.exclude_zero_kernel_dist:
            for k in range(2):
                if state_neighbour[k] < self.min_kernel_dist:
                    state_neighbour[k] = self.min_kernel_dist
                # alternatively:
                # state_neighbour[k] = np.max(state_neighbour[k],
                #                             self.min_kernel_dist)

        if self.ucm_wrapper.base_args["cc_method"] == "factors":
            # rescale so that the three weights add up to one
            weight_sum = sum(state_neighbour[2:])
            for k in range(2, 5):
                state_neighbour[k] /= weight_sum

        # update the state
        self.state = state_neighbour

    def energy(self):
        """Compute the state's value of the metric to optimize (minimize)."""
        ucm_args = self._ucm_params_dict.copy()
        comparison_df = self.ucm_wrapper.get_sample_comparison_df(ucm_args=ucm_args)

        return self.compute_metric(comparison_df["obs"], comparison_df["pred"])

    def calibrate(self, *, initial_solution=None, num_steps=None, num_update_logs=None):
        """
        Calibrate the urban cooling model for the given data.

        Run a simulated annealing procedure to get the arguments of the InVEST urban
        cooling model that minimize the performance metric.

        Parameters
        ----------
        initial_solution : list-like, optional
            Sequence with the parameter values used as initial solution, of the form
            (t_air_average_radius, green_area_cooling_distance, cc_weight_shade,
            cc_weight_albedo, cc_weight_eti) when the cooling capacity method is
            "factors", or (t_air_average_radius, green_area_cooling_distance) when the
            method is "intensity". If not provided, the default values of the urban
            cooling model will be used.
        num_steps : int, optional.
            Number of iterations of the simulated annealing procedure. If not provided,
            the value set in `settings.DEFAULT_NUM_STEPS` will be used.
        num_update_logs : int, default 100
            Number of updates that will be logged. If `num_steps` is equal to
            `num_update_logs`, each iteration will be logged. If not provided, the value
            set in `settings.DEFAULT_UPDATE_LOGS` will be used.

        Returns
        -------
        (state, metric) : the best state, i.e., combination of model parameters and the
            corresponding metric.
        """
        # Override the values set in the init method. Note that the attribute names are
        # defined in the `Annealer` class
        if initial_solution is not None:
            self.state = initial_solution
        if num_steps is not None:
            self.steps = num_steps
        if num_update_logs is not None:
            self.updates = num_update_logs

        return self.anneal()

    # shortcuts to useful `UCMWrapper` methods
    # TODO: dry `ucm_args` with a decorator?
    def predict_t_arr(self, i, *, ucm_args=None):
        """
        Predict a temperature array for one of the calibration dates.

        Parameters
        ----------
        i : int
            Positional index of the calibration date.
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of the urban
            cooling model. The provided keys will override those set in the `base_args`
            attribute of this class (set up in the initialization method).

        Returns
        -------
        t_arr : np.ndarray
            Predicted temperature array aligned with the LULC raster for the selected
            date.
        """
        if ucm_args is None:
            ucm_args = self._ucm_params_dict.copy()

        return self.ucm_wrapper.predict_t_arr(i, ucm_args=ucm_args)

    def predict_t_da(self, *, ucm_args=None):
        """
        Predict a temperature data-array.

        The array is aligned with the LULC raster for all the calibration dates.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of the urban
            cooling model. The provided keys will override those set in the current
            solution found by the calibrator, i.e., the `state` attribute.

        Returns
        -------
        t_da : xr.DataArray
            Predicted temperature data array aligned with the LULC raster.
        """
        if ucm_args is None:
            ucm_args = self._ucm_params_dict.copy()

        return self.ucm_wrapper.predict_t_da(ucm_args=ucm_args)

    def get_sample_comparison_df(self, *, ucm_args=None):
        """
        Compute a comparison data frame of the observed and predicted values.

        Each row corresponds to a sample (i.e., station measurement for a specific
        date). Requires that the object has been instantiated with either
        `t_raster_filepath` or `station_t_filepath`.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of the urban
            cooling model. The provided keys will override those set in the current
            solution found by the calibrator, i.e., the `state` attribute.

        Returns
        -------
        sample_comparison_df : pd.DataFrame
            Comparison data frame with columns for the sample date, station, observed
            and predicted values.
        """
        if ucm_args is None:
            ucm_args = self._ucm_params_dict.copy()

        return self.ucm_wrapper.get_sample_comparison_df(ucm_args=ucm_args)

    def get_model_perf_df(self, *, ucm_args=None, num_runs=None):
        """
        Compute a model performance data frame.

        Compare the performance of the calibrated model with randomly sampling
        temperature values from the :math:`[T_{ref}, T_{ref} + UHI_{max}]` range
        according to a uniform and normal distribution. Requires that the object has
        been instantiated with either `t_raster_filepath` or `station_t_filepath`.

        Parameters
        ----------
        ucm_args : dict-like, optional
            Custom keyword arguments to be passed to the `execute` method of the urban
            cooling model. The provided keys will override those set in the current
            solution found by the calibrator, i.e., the `state` attribute.
        num_runs : int, optional
            Number of runs over which the results of randomly sampling (from both the
            uniform and normal distribution) will be averaged. If not provided, the
            value set in `settings.DEFAULT_MODEL_PERF_NUM_RUNS` will be used.

        Returns
        -------
        model_perf_df : pd.DataFrame
            Predicted temperature data array aligned with the LULC raster.
        """
        if ucm_args is None:
            ucm_args = self._ucm_params_dict.copy()

        return self.ucm_wrapper.get_model_perf_df(ucm_args=ucm_args, num_runs=num_runs)
