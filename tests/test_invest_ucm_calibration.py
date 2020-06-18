import glob
import os
import shutil
import unittest
from os import path

from click import testing

import invest_ucm_calibration as iuc
from invest_ucm_calibration.cli import main


class TestIUC(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'tests/data'

        self.lulc_raster_filepath = path.join(self.data_dir, 'lulc.tif')
        self.biophysical_table_filepath = path.join(self.data_dir,
                                                    'biophysical-table.csv')
        self.ref_et_raster_filepaths = glob.glob(
            path.join(self.data_dir, 'ref_et*.tif'))

        # calibrate with temperature map
        self.t_raster_filepaths = glob.glob(path.join(self.data_dir, 'T*.tif'))

        # calibrate with station measurements
        self.station_t_filepath = path.join(self.data_dir, 'station-t.csv')
        self.station_locations_filepath = path.join(self.data_dir,
                                                    'station-locations.csv')
        self.station_t_one_day_filepath = path.join(self.data_dir,
                                                    'station-t-one-day.csv')

        # other parameters
        self.cc_method = 'factors'
        self.num_steps = 2
        self.num_update_logs = 2
        # self.workspace_dir = path.join(self.data_dir, 'tmp')
        # os.mkdir(self.workspace_dir)

    # def tearDown(self):
    #     shutil.rmtree(self.workspace_dir)

    def test_one_day(self):
        ref_et_raster_filepath = self.ref_et_raster_filepaths[0]
        t_raster_filepath = self.t_raster_filepaths[0]
        t_refs = 20
        uhi_maxs = 10

        # calibrate with map
        # no t_refs/no uhi_maxs
        iuc.UCMCalibrator(self.lulc_raster_filepath,
                          self.biophysical_table_filepath, self.cc_method,
                          ref_et_raster_filepath,
                          t_raster_filepaths=t_raster_filepath,
                          num_steps=self.num_steps,
                          num_update_logs=self.num_update_logs)
        # no t_refs
        iuc.UCMCalibrator(self.lulc_raster_filepath,
                          self.biophysical_table_filepath, self.cc_method,
                          ref_et_raster_filepath, uhi_maxs=uhi_maxs,
                          t_raster_filepaths=t_raster_filepath,
                          num_steps=self.num_steps,
                          num_update_logs=self.num_update_logs)
        # no uhi_maxs
        iuc.UCMCalibrator(self.lulc_raster_filepath,
                          self.biophysical_table_filepath, self.cc_method,
                          ref_et_raster_filepath, t_refs=t_refs,
                          t_raster_filepaths=t_raster_filepath,
                          num_steps=self.num_steps,
                          num_update_logs=self.num_update_logs)
        # both t_refs/uhi_maxs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, ref_et_raster_filepath, t_refs=t_refs,
            uhi_maxs=uhi_maxs, t_raster_filepaths=t_raster_filepath,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)

        # calibrate with measurements
        # no t_refs/no uhi_maxs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, ref_et_raster_filepath,
            station_t_filepath=self.station_t_one_day_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)
        # no t_refs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, ref_et_raster_filepath, uhi_maxs=uhi_maxs,
            station_t_filepath=self.station_t_one_day_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)
        # no uhi_maxs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, ref_et_raster_filepath, t_refs=t_refs,
            station_t_filepath=self.station_t_one_day_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)
        # both t_refs/uhi_maxs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, ref_et_raster_filepath, t_refs=t_refs,
            uhi_maxs=uhi_maxs,
            station_t_filepath=self.station_t_one_day_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)

    def test_multiple_days(self):
        t_refs = [20, 21]
        uhi_maxs = [10, 11]

        # print(path.exists(t_raster_filepath))
        # calibrate with map
        # no t_refs/no uhi_maxs
        iuc.UCMCalibrator(self.lulc_raster_filepath,
                          self.biophysical_table_filepath, self.cc_method,
                          self.ref_et_raster_filepaths,
                          t_raster_filepaths=self.t_raster_filepaths,
                          num_steps=self.num_steps,
                          num_update_logs=self.num_update_logs)
        # no t_refs
        iuc.UCMCalibrator(self.lulc_raster_filepath,
                          self.biophysical_table_filepath, self.cc_method,
                          self.ref_et_raster_filepaths, uhi_maxs=uhi_maxs,
                          t_raster_filepaths=self.t_raster_filepaths,
                          num_steps=self.num_steps,
                          num_update_logs=self.num_update_logs)
        # no uhi_maxs
        iuc.UCMCalibrator(self.lulc_raster_filepath,
                          self.biophysical_table_filepath, self.cc_method,
                          self.ref_et_raster_filepaths, t_refs=t_refs,
                          t_raster_filepaths=self.t_raster_filepaths,
                          num_steps=self.num_steps,
                          num_update_logs=self.num_update_logs)
        # both t_refs/uhi_maxs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, self.ref_et_raster_filepaths, t_refs=t_refs,
            uhi_maxs=uhi_maxs, t_raster_filepaths=self.t_raster_filepaths,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)

        # calibrate with measurements
        # no t_refs/no uhi_maxs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, self.ref_et_raster_filepaths,
            station_t_filepath=self.station_t_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)
        # no t_refs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, self.ref_et_raster_filepaths, uhi_maxs=uhi_maxs,
            station_t_filepath=self.station_t_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)
        # no uhi_maxs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, self.ref_et_raster_filepaths, t_refs=t_refs,
            station_t_filepath=self.station_t_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)
        # both t_refs/uhi_maxs
        iuc.UCMCalibrator(
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, self.ref_et_raster_filepaths, t_refs=t_refs,
            uhi_maxs=uhi_maxs, station_t_filepath=self.station_t_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps, num_update_logs=self.num_update_logs)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'tests/data'

        self.lulc_raster_filepath = path.join(self.data_dir, 'lulc.tif')
        self.biophysical_table_filepath = path.join(self.data_dir,
                                                    'biophysical-table.csv')
        self.ref_et_raster_filepaths = glob.glob(
            path.join(self.data_dir, 'ref_et*.tif'))

        # calibrate with temperature map
        self.t_raster_filepaths = glob.glob(path.join(self.data_dir, 'T*.tif'))

        # calibrate with station measurements
        self.station_t_filepath = path.join(self.data_dir, 'station-t.csv')
        self.station_locations_filepath = path.join(self.data_dir,
                                                    'station-locations.csv')
        self.station_t_one_day_filepath = path.join(self.data_dir,
                                                    'station-t-one-day.csv')

        # other parameters
        self.cc_method = 'factors'
        self.num_steps = 2
        self.num_update_logs = 2
        self.workspace_dir = path.join(self.data_dir, 'tmp')
        os.mkdir(self.workspace_dir)

        self.runner = testing.CliRunner()

        # TODO: test more possibilities of `args` in `invoke`
        # TODO: test `_dict_from_kws`

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_one_day(self):
        ref_et_raster_filepath = self.ref_et_raster_filepaths[0]
        t_raster_filepath = self.t_raster_filepaths[0]
        t_refs = 20
        uhi_maxs = 10

        # calibrate with map
        result = self.runner.invoke(main.cli, [
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, '--ref-et-raster-filepaths',
            ref_et_raster_filepath, '--t-refs', t_refs, '--uhi-maxs', uhi_maxs,
            '--t-raster-filepaths', t_raster_filepath, '--num-steps', 1,
            '--num-update-logs', 1, '--dst-filepath',
            path.join(self.workspace_dir, 'foo.json')
        ])
        self.assertEqual(result.exit_code, 0)

        # calibrate with measurements
        result = self.runner.invoke(main.cli, [
            self.lulc_raster_filepath, self.biophysical_table_filepath,
            self.cc_method, '--ref-et-raster-filepaths',
            ref_et_raster_filepath, '--t-refs', t_refs, '--uhi-maxs', uhi_maxs,
            '--station-t-filepath', self.station_t_one_day_filepath,
            '--station-locations-filepath', self.station_locations_filepath,
            '--num-steps', 1, '--num-update-logs', 1, '--dst-filepath',
            path.join(self.workspace_dir, 'bar.json')
        ])
        self.assertEqual(result.exit_code, 0)

    # def test_multiple_days(self):
    #     t_refs = [20, 21]
    #     uhi_maxs = [10, 11]

    #     # calibrate with map
    #     result = self.runner.invoke(main.cli, [
    #         self.lulc_raster_filepath, self.biophysical_table_filepath,
    #         self.aoi_vector_filepath, self.cc_method,
    #         self.ref_et_raster_filepaths, t_refs, uhi_maxs,
    #         self.t_raster_filepaths, None, None, None, None, None, None,
    #         None, None, None, 1, 1
    #     ])
    #     self.assertEqual(result.exit_code, 0)

    #     # calibrate with measurements
    #     result = self.runner.invoke(main.cli, [
    #         self.lulc_raster_filepath, self.biophysical_table_filepath,
    #         self.aoi_vector_filepath, self.cc_method,
    #         self.ref_et_raster_filepaths, t_refs, uhi_maxs, None,
    #         self.station_t_filepath, self.station_locations_filepath, None,
    #         None, None, None, None, None, None, 1, 1
    #     ])
    #     self.assertEqual(result.exit_code, 0)

    # def test_other_args(self):
    #     cc_methods = ['factors', 'intensity']
    #     # workspace_dir
    #     # initial_solution
    #     # extra_ucm_args
    #     metrics = ['R2', 'MAE', 'RMSE']
    #     stepsize = .3
    #     num_workers = 10
    #     num_steps = 10
    #     num_update_logs = 10
