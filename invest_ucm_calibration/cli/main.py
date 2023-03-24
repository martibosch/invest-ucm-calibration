"""CLI."""
import json
import logging
import warnings
from os import path
from typing import Any, Dict, List, Optional, Union

import fire

import invest_ucm_calibration as iuc
from invest_ucm_calibration import settings


def _process_sequence_arg(sequence_arg):
    """
    Process a (potentially) sequence CLI argument.

    Such an argument can either be a single value or a list of values (provided between
    quotes in the CLI and with values separated either between commas or spaces).
    """
    # check if the argument is a string
    if isinstance(sequence_arg, str):
        # check if the `sequence_arg` corresponds to a single value or to a list of values
        if "," in sequence_arg:
            # if the `sequence_arg` contains a comma, we assume that it corresponds to a
            # comma-separated list of values
            sequence = sequence_arg.split(",")
        elif " " in sequence_arg:
            # if the `sequence_arg` does not contain a comma but contains a space, we
            # assume that it corresponds to a space-separated list of values
            sequence = sequence_arg.split(" ")
        else:
            # otherwise, we assume that it corresponds to a single value
            sequence = [sequence_arg]
    else:
        # we assume that the argument is numeric and return it as a list
        sequence = [sequence_arg]
    return sequence


def _process_existing_filepath(filepath: str):
    """Check if the provided filepath exists."""
    if not path.exists(filepath):
        raise FileNotFoundError(filepath)


def _process_filepaths_arg(filepaths_arg: str, *, exists: bool = False) -> List[str]:
    """
    Process a filepath(s) CLI argument.

    Check if the provided file path string corresponds to a single file or to a list of
    files, potentially checking if the files exist.

    Parameters
    ----------
    filepaths_arg : str
        A string containing a single file path or a list of file paths, separated by
        either a comma or a space character (in the latter case, the file paths must not
        contain spaces).
    exists : bool, optional, default False
        Whether to check if the file(s) exist.

    Returns
    -------
    filepaths : list
    """
    filepaths = _process_sequence_arg(filepaths_arg)

    # check if the files exist
    if exists:
        for filepath in filepaths:
            _process_existing_filepath(filepath)
    return filepaths


def cli(
    lulc_raster_filepath: str,
    biophysical_table_filepath: str,
    cc_method: str,
    ref_et_raster_filepaths: str,
    aoi_vector_filepath: Optional[str] = None,
    t_refs: Optional[Union[float, str]] = None,
    uhi_maxs: Optional[Union[float, str]] = None,
    t_raster_filepaths: Optional[str] = None,
    station_t_filepath: Optional[str] = None,
    station_locations_filepath: Optional[str] = None,
    dates: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    initial_solution: Optional[str] = None,
    extra_ucm_args: Optional[str] = None,
    metric: Optional[str] = None,
    stepsize: Optional[float] = None,
    exclude_zero_kernel_dist: Optional[bool] = True,
    num_steps: Optional[int] = None,
    num_update_logs: Optional[int] = None,
    dst_filepath: Optional[str] = None,
):
    """
    Calibrate the InVEST urban cooling model.

    Parameters
    ----------
    lulc_raster_filepath : str
        Path to the land use/land cover raster.
    biophysical_table_filepath : str
        Path to the biophysical table.
    cc_method : str
        Calibration method. Must be one of "factors" or "intensity".
    ref_et_raster_filepaths : str
        Path to the reference evapotranspiration raster, or sequence of paths to
        evapotranspiration rasters. If providing a sequence of paths, they must be
        enclosed by quotes and separated by either a comma or a space character (in the
        latter case, the file paths must not contain spaces).
    aoi_vector_filepath : str, optional
        Path to the area of interest vector. If not provided, the bounds of the LULC
        raster will be used.
    t_refs : numeric or str, optional
        Reference air temperature or sequence of reference air temperatures. If
        providing a sequence of values, they must be enclosed by quotes and separated by
        either a comma or a space character. If not provided, it will be set as the
        minimum observed temperature (raster or station measurements, for each
        respective date if calibrating for multiple dates).
    uhi_maxs : numeric or str, optional
        Magnitude of the UHI effect or sequence of UHI effect magnitudes. If not
        provided, it will be set as the difference between the maximum and minimum
        observed temperature (raster or station measurements, for each respective date
        if calibrating for multiple dates).
    t_raster_filepaths : str, optional
        Path to the observed temperature raster, or sequence of strings of path to the
        observed temperature rasters. If providing a sequence of paths, they must be
        enclosed by quotes and separated by either a comma or a space character (in the
        latter case, the file paths must not contain spaces). Required if calibrating
        against temperature rasters. If not provided, `station_t_filepath` and
        `station_locations_filepath` must be provided.
    station_t_filepath : str, optional
        Path to a table of air temperature measurements where each column corresponds to
        a monitoring station and each row to a datetime. Required alongside
        `station_locations_filepath` if calibrating against station measurements.
        Otherwise, `t_raster_filepaths` must be provided.
    station_locations_filepath : str, optional
        Path to a table with the locations of each monitoring station, where the first
        column features the station labels (that match the columns of the table of air
        temperature measurements), and there are (at least) a column labelled `x` and a
        column labelled `y` that correspod to the locations of each station (in the same
        CRS as the other rasters). Required alongside `station_t_filepath` if
        calibrating against station measurements. Otherwise, `t_raster_filepaths` must
        be provided.
    dates : str, optional
        Date or list of dates that correspond to each of the observed temperature raster
        provided in t_raster_filepaths. Ignored if `station_t_filepath` is provided.
    workspace_dir : str, optional
        Path to the folder where the model outputs will be written. If not provided, a
        temporary directory will be used.
    initial_solution : str, optional
        Sequence with the parameter values used as initial solution, which can either be
        of the form (t_air_average_radius, green_area_cooling_distance, cc_weight_shade,
        cc_weight_albedo, cc_weight_eti) when `cc_method` is "factors", or
        (t_air_average_radius, green_area_cooling_distance) when `cc_method` is
        "intensity". If not provided, the default values of the urban cooling model will
        be used.
    extra_ucm_args : str, optional
        Other keyword arguments to be passed to the `execute` method of the urban
        cooling model, as a sequence of "key:value" pairs, enclosed by quotes and
        separated by either a comma or a space character.
    metric : str, optional
        Target metric to optimize in the calibration. Can be either "R2" for the R
        squared (which will be maximized), "MAE" for the mean absolute error (which will
        be minimized) or "RMSE" for the (root) mean squared error (which will be
        minimized). If not provided, the value set in `settings.DEFAULT_METRIC` will be
        used.
    stepsize : numeric, optional
        Step size in terms of the fraction of each parameter when looking to select a
        neighbor solution for the following iteration. The neighbor will be randomly
        drawn from an uniform distribution in the [param - stepsize * param, param +
        stepsize * param] range. For example, with a step size of 0.3 and a
        `t_air_average_radius` of 500 at a given iteration, the solution for the next
        iteration will be uniformly sampled from the [350, 650] range. If not provided,
        the value set in `settings.DEFAULT_STEPSIZE` will be used.
    exclude_zero_kernel_dist : bool, optional
        Whether the calibration should consider parameters that lead to decay functions
        with a kernel distance of zero pixels (i.e., `t_air_average_radius` or
        `green_area_cooling_distance` lower than half the LULC pixel resolution).
    num_steps : int, optional
        Number of iterations of the simulated annealing procedure. If not provided, the
        value set in `settings.DEFAULT_NUM_STEPS` will be used.
    num_update_logs : int, optional
        Number of updates that will be logged. If `num_steps` is equal to
        `num_update_logs`, each iteration will be logged. If not provided, the value set
        in `settings.DEFAULT_UPDATE_LOGS` will be used.
    dst_filepath : str, optional
        Path to dump the calibrated parameters. If not provided, no file will be created
        (nonetheless, the calibrated parameters will be logged).
    """
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # disable InVEST's logging
    for module in (
        "natcap.invest.urban_cooling_model",
        "natcap.invest.utils",
        "pygeoprocessing.geoprocessing",
        "taskgraph.Task",
    ):
        logging.getLogger(module).setLevel(logging.WARNING)
    # ignore all warnings
    warnings.filterwarnings("ignore")

    # preprocess (potentially) sequence arguments
    ref_et_raster_filepaths = _process_filepaths_arg(
        ref_et_raster_filepaths, exists=True
    )  # type: ignore
    if t_refs is not None:
        t_refs = [float(val) for val in _process_sequence_arg(t_refs)]  # type: ignore
    if uhi_maxs is not None:
        uhi_maxs = [
            float(val) for val in _process_sequence_arg(uhi_maxs)  # type: ignore
        ]
    if t_raster_filepaths is not None:
        t_raster_filepaths = _process_filepaths_arg(
            t_raster_filepaths, exists=True
        )  # type: ignore
    if dates is not None:
        dates = _process_sequence_arg(dates)  # type: ignore

    # process single filepath arguments to check that they exist if provided
    for filepath in (
        lulc_raster_filepath,
        biophysical_table_filepath,
    ):
        _process_existing_filepath(filepath)  # type: ignore
    for filepath in (
        aoi_vector_filepath,
        station_t_filepath,
        station_locations_filepath,
    ):
        if filepath is not None:
            _process_existing_filepath(filepath)  # type: ignore

    # instantiate the calibration object
    ucm_calibrator = iuc.UCMCalibrator(
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
        initial_solution=initial_solution,
        extra_ucm_args=extra_ucm_args,
        metric=metric,
        stepsize=stepsize,
        exclude_zero_kernel_dist=exclude_zero_kernel_dist,
        num_steps=num_steps,
        num_update_logs=num_update_logs,
    )
    # solution, cost = ucm_calibrator.calibrate()
    solution, cost = ucm_calibrator.anneal()
    logger.info("Best solution %s with cost %s", str(solution), cost)

    if dst_filepath:
        # TODO: dry and reuse ucm_calibrator attributes
        param_keys = list(settings.DEFAULT_UCM_PARAMS)
        if cc_method == "factors":
            param_keys += list(settings.DEFAULT_UCM_FACTORS_PARAMS)
        with open(dst_filepath, "w") as dst:
            json.dump(
                {
                    param_key: param_value
                    for param_key, param_value in zip(param_keys, solution, strict=True)
                },
                dst,
            )
        logger.info("Dumped calibrated parameters to %s", dst_filepath)


def main():
    """Entrypoint."""
    fire.Fire(cli)
