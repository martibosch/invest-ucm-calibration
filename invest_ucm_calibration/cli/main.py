import json
import logging
import warnings

import click

import invest_ucm_calibration as iuc
from invest_ucm_calibration import settings


# utils for the CLI
class OptionEatAll(click.Option):
    # Option that can take an unlimided number of arguments
    # Copied from Stephen Rauch's answer in stack overflow.
    # https://bit.ly/2kstLhe
    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(
                name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


def _dict_from_kws(kws):
    # Multiple key:value pair arguments in click, see https://bit.ly/32BaES3
    if kws is not None:
        kws = dict(kw.split(':') for kw in kws)
    else:
        kws = {}

    return kws


# CLI
@click.command()
@click.argument('lulc_raster_filepath', type=click.Path(exists=True))
@click.argument('biophysical_table_filepath', type=click.Path(exists=True))
@click.argument('cc_method')
@click.option(
    '--ref-et-raster-filepaths', cls=OptionEatAll, required=True,
    help='Path to the reference evapotranspiration raster, or sequence of '
    'strings with a path to the reference evapotranspiration raster')
@click.option(
    '--t-refs', cls=OptionEatAll,
    help='Reference air temperature. If not provided, it will be set as the '
    'minimum observed temperature (raster or station measurements, for each '
    'respective date if calibrating for multiple dates).')
@click.option(
    '--uhi-maxs', cls=OptionEatAll,
    help='Magnitude of the UHI effect. If not provided, it will be set as the '
    'difference between the maximum and minimum observed temperature (raster '
    'or station measurements, for each respective date if calibrating for '
    'multiple dates).')
@click.option(
    '--t-raster-filepaths', cls=OptionEatAll,
    help='Path to the observed temperature raster, or sequence of strings with'
    ' a path to the observed temperature rasters. The raster must be aligned '
    'to the LULC raster. Required if calibrating against temperature map(s).')
@click.option(
    '--station-t-filepath', type=click.Path(exists=True),
    help='Path to a table of air temperature measurements where each column '
    'corresponds to a monitoring station and each row to a datetime. Required '
    'if calibrating against station measurements.')
@click.option(
    '--station-locations-filepath', type=click.Path(exists=True),
    help='Path to a table with the locations of each monitoring station, where'
    ' the first column features the station labels (that match the columns of '
    'the table of air temperature measurements), and there are (at least) a '
    'column labelled `x` and a column labelled `y` that correspod to the '
    'locations of each station (in the same CRS as the other rasters). '
    'Required if calibrating against station measurements.')
@click.option(
    '--dates', cls=OptionEatAll,
    help='Date or list of dates that correspond to each of the observed '
    'temperature raster provided in t_raster_filepaths. Ignored if '
    '`station_t_filepath` is provided.')
@click.option(
    '--align-rasters/--no-align-rasters', default=True,
    help='Whether the rasters should be aligned before passing them as '
    'arguments of the InVEST urban cooling model. Since the model already '
    'aligns the LULC and reference evapotranspiration rasters, this argument '
    'is only useful to align the temperature rasters, and is therefore ignored'
    ' if calibrating against station measurements.')
@click.option(
    '--workspace-dir', type=click.Path(exists=True),
    help='Path to the folder where the model outputs will be written. If not '
    'provided, a temporary directory will be used.')
@click.option(
    '--initial-solution', cls=OptionEatAll,
    help='Sequence with the parameter values used as initial solution, of the '
    'form (t_air_average_radius, green_area_cooling_distance, cc_weight_shade,'
    ' cc_weight_albedo, cc_weight_eti). If not provided, the default values of'
    ' the urban cooling model will be used.')
@click.option(
    '--extra-ucm-args', cls=OptionEatAll,
    help='Other keyword arguments to be passed to the `execute` method of the '
    'urban cooling model, as a sequence of "key:value" pairs')
@click.option(
    '--metric',
    help='Target metric to optimize in the calibration. Can be either `R2` for'
    ' the R squared (which will be maximized), `MAE` for the mean absolute '
    'error (which will be minimized) or `RMSE` for the (root) mean squared '
    'error (which will be minimized). If not provided, the value set in '
    '`settings.DEFAULT_METRIC` will be used.')
@click.option(
    '--stepsize', type=float,
    help='Step size in terms of the fraction of each parameter when looking to'
    'select a neighbor solution for the following iteration. The neighbor will'
    ' be randomly drawn from an uniform distribution in the [param - stepsize '
    '* param, param + stepsize * param] range. For example, with a step size '
    'of 0.3 and a `t_air_average_radius` of 500 at a given iteration, the '
    'solution for the next iteration will be uniformly sampled from the [350, '
    '650] range. If not provided, the value set in `settings.DEFAULT_STEPSIZE`'
    ' will be used.')
@click.option(
    '--exclude-zero-kernel-dist/--no-exclude-zero-kernel-dist', default=True,
    help='Whether the calibration should consider parameters that lead to '
    'decay functions with a kernel distance of zero pixels (i.e., '
    '`t_air_average_radius` or `green_area_cooling_distance` lower than half '
    'the LULC pixel resolution).')
@click.option(
    '--num-workers', type=int,
    help='Number of workers so that the simulations of each iteration can be '
    'executed at scale. Only useful if calibrating for multiple dates. If not '
    'provided, it will be set automatically depending on the number of dates '
    'and available number of processors in the CPU.')
@click.option(
    '--num-steps', type=int,
    help='Number of iterations of the simulated annealing procedure. If not '
    'provided, the value set in `settings.DEFAULT_NUM_STEPS` will be used.')
@click.option(
    '--num-update-logs', type=int,
    help='Number of updates that will be logged. If `num_steps` is equal to '
    '`num_update_logs`, each iteration will be logged. If not provided, the '
    'value set in `settings.DEFAULT_UPDATE_LOGS` will be used.')
@click.option(
    '--dst-filepath', type=click.Path(), required=True,
    help='Path to dump the calibrated parameters. If not provided, no file '
    'will be created (nonetheless, the calibrated parameters will be logged)')
def cli(lulc_raster_filepath, biophysical_table_filepath, cc_method,
        ref_et_raster_filepaths, t_refs, uhi_maxs, t_raster_filepaths,
        station_t_filepath, station_locations_filepath, dates, align_rasters,
        workspace_dir, initial_solution, extra_ucm_args, metric, stepsize,
        exclude_zero_kernel_dist, num_workers, num_steps, num_update_logs,
        dst_filepath):
    """
    Calibrate the InVEST urban cooling model

    Arguments
    ----------
    lulc_raster_filepath : str
        Path to the raster of land use/land cover (LULC) file
    biophysical_table_filepath : str
        Path to the biophysical table CSV file
    cc_method : str
        Cooling capacity calculation method. Can be either 'factors' or
        'intensity'
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # disable InVEST's logging
    for module in ('natcap.invest.urban_cooling_model', 'natcap.invest.utils',
                   'pygeoprocessing.geoprocessing'):
        logging.getLogger(module).setLevel(logging.WARNING)
    # ignore all warnings
    warnings.filterwarnings('ignore')

    # preprocess extra args to the urban cooling model. Transform values
    # (strings) to numeric when appropriate
    extra_ucm_args = _dict_from_kws(extra_ucm_args)
    for arg_key in extra_ucm_args:
        try:
            arg_val = float(extra_ucm_args[arg_key])
            extra_ucm_args[arg_key] = arg_val
        except ValueError:
            pass

    ucm_calibrator = iuc.UCMCalibrator(
        lulc_raster_filepath, biophysical_table_filepath, cc_method,
        ref_et_raster_filepaths, t_refs=t_refs, uhi_maxs=uhi_maxs,
        t_raster_filepaths=t_raster_filepaths,
        station_t_filepath=station_t_filepath,
        station_locations_filepath=station_locations_filepath, dates=dates,
        align_rasters=align_rasters, workspace_dir=workspace_dir,
        initial_solution=initial_solution, extra_ucm_args=extra_ucm_args,
        metric=metric, stepsize=stepsize,
        exclude_zero_kernel_dist=exclude_zero_kernel_dist,
        num_workers=num_workers, num_steps=num_steps,
        num_update_logs=num_update_logs)
    # solution, cost = ucm_calibrator.calibrate()
    solution, cost = ucm_calibrator.anneal()
    logger.info("Best solution %s with cost %s", str(solution), cost)

    if dst_filepath:
        with open(dst_filepath, 'w') as dst:
            json.dump(
                {
                    param_key: param_value
                    for param_key, param_value in zip(
                        settings.DEFAULT_UCM_PARAMS, solution)
                }, dst)
        logger.info("Dumped calibrated parameters to %s", dst_filepath)
