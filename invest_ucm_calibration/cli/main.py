import json
import logging
import warnings

import click

import invest_ucm_calibration as iuc


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
@click.argument('aoi_vector_filepath', type=click.Path(exists=True))
@click.argument('cc_method')
@click.option('--ref-et-raster-filepaths', cls=OptionEatAll, required=True)
@click.option('--t-refs', cls=OptionEatAll)
@click.option('--uhi-maxs', cls=OptionEatAll)
@click.option('--t-raster-filepaths', cls=OptionEatAll)
@click.option('--station-t-filepath', type=click.Path(exists=True))
@click.option('--station-locations-filepath', type=click.Path(exists=True))
@click.option('--workspace-dir', type=click.Path(exists=True))
@click.option('--initial-solution', cls=OptionEatAll)
@click.option('--extra-ucm-args', cls=OptionEatAll)
@click.option('--metric')
@click.option('--stepsize', type=float)
@click.option('--num-workers', type=int)
@click.option('--num-steps', type=int)
@click.option('--num-update-logs', type=int)
@click.option('--dst-filepath', type=click.Path())
def cli(lulc_raster_filepath, biophysical_table_filepath, aoi_vector_filepath,
        cc_method, ref_et_raster_filepaths, t_refs, uhi_maxs,
        t_raster_filepaths, station_t_filepath, station_locations_filepath,
        workspace_dir, initial_solution, extra_ucm_args, metric, stepsize,
        num_workers, num_steps, num_update_logs, dst_filepath):
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
        lulc_raster_filepath, biophysical_table_filepath, aoi_vector_filepath,
        cc_method, ref_et_raster_filepaths, T_refs=t_refs, uhi_maxs=uhi_maxs,
        T_raster_filepaths=t_raster_filepaths,
        station_T_filepath=station_t_filepath,
        station_locations_filepath=station_locations_filepath,
        workspace_dir=workspace_dir, initial_solution=initial_solution,
        extra_ucm_args=extra_ucm_args, metric=metric, stepsize=stepsize,
        num_workers=num_workers, num_steps=num_steps,
        num_update_logs=num_update_logs)
    solution, cost = ucm_calibrator.anneal()
    logger.info("Best solution %s with cost %s", str(solution), cost)

    if dst_filepath:
        with open(dst_filepath, 'w') as dst:
            json.dump(
                {
                    param_key: param_value
                    for param_key, param_value in zip(
                        ucm_calibrator.DEFAULT_MODEL_PARAMS, solution)
                }, dst)
        logger.info("Dumped calibrated parameters to %s", dst_filepath)
