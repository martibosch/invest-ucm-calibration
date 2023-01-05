import json
import logging
import warnings
from typing import Any, Dict, List, Optional

import fire

import invest_ucm_calibration as iuc
from invest_ucm_calibration import settings


def cli(
    lulc_raster_filepath: str,
    biophysical_table_filepath: str,
    cc_method: str,
    ref_et_raster_filepaths: List[str],
    t_refs: Optional[List[float]] = None,
    uhi_maxs: Optional[List[float]] = None,
    t_raster_filepaths: Optional[List[str]] = None,
    station_t_filepath: Optional[str] = None,
    station_locations_filepath: Optional[str] = None,
    dates: Optional[List[str]] = None,
    workspace_dir: Optional[str] = None,
    initial_solution: Optional[List[float]] = None,
    extra_ucm_args: Optional[Dict[str, Any]] = None,
    metric: Optional[str] = "mean_absolute_error",
    stepsize: Optional[float] = None,
    exclude_zero_kernel_dist: Optional[bool] = True,
    num_workers: Optional[int] = None,
    num_steps: Optional[int] = None,
    num_update_logs: Optional[int] = None,
    dst_filepath: Optional[str] = None,
):
    """
    Calibrates the urban cooling model (UCM) using the provided data.

    Parameters
    ----------
    lulc_raster_filepath : str
        Path to the land use/land cover raster.
    biophysical_table_filepath : str
        Path to the biophysical table.
    cc_method : str
        Calibration method. Must be one of "measured", "simulated", or
        "measured_and_simulated".
    ref_et_raster_filepaths : List[str]
        Path to the reference evapotranspiration raster, or sequence of strings with a
        path to the reference evapotranspiration raster.
    t_refs : Optional[List[float]], optional
        Reference air temperature. If not provided, it will be set as the minimum
        observed temperature (raster or station measurements, for each respective date
        if calibrating for multiple dates).
    uhi_maxs : Optional[List[float]], optional
        Magnitude of the UHI effect. If not provided, it will be set as the difference
        between the maximum and minimum observed temperature (raster or station
        measurements, for each respective date if calibrating for multiple dates).
    t_raster_filepaths : Optional[List[str]], optional
        Path to the observed temperature raster, or sequence of strings with a path to
        the observed temperature rasters.
    metric : Optional[str], optional
        Target metric to optimize in the calibration. Can be either `R2` for the R
        squared (which will be maximized), `MAE` for the mean absolute error (which will
        be minimized) or `RMSE` for the (root) mean squared error (which will be
        minimized). If not provided, the value set in `settings.DEFAULT_METRIC` will be
        used.
    stepsize : Optional[float], optional
        Step size in terms of the fraction of each parameter when looking to select a
        neighbor solution for the following iteration. The neighbor will be randomly
        drawn from an uniform distribution in the [param - stepsize * param, param +
        stepsize * param] range. For example, with a step size of 0.3 and a
        `t_air_average_radius` of 500 at a given iteration, the solution for the next
        iteration will be uniformly sampled from the [350, 650] range. If not provided,
        the value set in `settings.DEFAULT_STEPSIZE` will be used.
    exclude_zero_kernel_dist : Optional[bool], optional
        Whether the calibration should consider parameters that lead to decay functions
        with a kernel distance of zero pixels (i.e., `t_air_average_radius` or
        `green_area_cooling_distance` lower than half the LULC pixel resolution).
    num_workers : Optional[int], optional
        Number of workers so that the simulations of each iteration can be executed at
        scale. Only useful if calibrating for multiple dates. If not provided, it will
        be set automatically depending on the number of dates and available number of
        processors in the CPU.
    num_steps : Optional[int], optional
        Number of iterations of the simulated annealing procedure. If not provided, the
        value set in `settings.DEFAULT_NUM_STEPS` will be used.
    num_update_logs : Optional[int], optional
        Number of updates that will be logged. If `num_steps` is equal to
        `num_update_logs`, each iteration will be logged. If not provided, the value set
        in `settings.DEFAULT_UPDATE_LOGS` will be used.
    """
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # disable InVEST's logging
    for module in (
        "natcap.invest.urban_cooling_model",
        "natcap.invest.utils",
        "pygeoprocessing.geoprocessing",
    ):
        logging.getLogger(module).setLevel(logging.WARNING)
    # ignore all warnings
    warnings.filterwarnings("ignore")

    ucm_calibrator = iuc.UCMCalibrator(
        lulc_raster_filepath,
        biophysical_table_filepath,
        cc_method,
        ref_et_raster_filepaths,
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
        num_workers=num_workers,
        num_steps=num_steps,
        num_update_logs=num_update_logs,
    )
    # solution, cost = ucm_calibrator.calibrate()
    solution, cost = ucm_calibrator.anneal()
    logger.info("Best solution %s with cost %s", str(solution), cost)

    if dst_filepath:
        with open(dst_filepath, "w") as dst:
            json.dump(
                {
                    param_key: param_value
                    for param_key, param_value in zip(
                        settings.DEFAULT_UCM_PARAMS, solution
                    )
                },
                dst,
            )
        logger.info("Dumped calibrated parameters to %s", dst_filepath)


def main():
    fire.Fire(cli)
