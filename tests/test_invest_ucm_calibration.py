"""Tests."""
import glob
import os
import shutil
import subprocess
import unittest
from os import path

import pytest

import invest_ucm_calibration as iuc
from invest_ucm_calibration.cli import main


def _encode_as_cli_arg(arg, sep):
    return f"'{sep.join([str(item) for item in arg])}'"


class TestIUC(unittest.TestCase):
    def setUp(self):
        self.data_dir = "tests/data"

        self.lulc_raster_filepath = path.join(self.data_dir, "lulc.tif")
        self.biophysical_table_filepath = path.join(
            self.data_dir, "biophysical-table.csv"
        )
        self.ref_et_raster_filepaths = glob.glob(
            path.join(self.data_dir, "ref_et*.tif")
        )

        # area of interest vector
        self.aoi_vector_filepath = path.join(self.data_dir, "aoi.gpkg")

        # calibrate with temperature map
        self.t_raster_filepaths = glob.glob(path.join(self.data_dir, "T*.tif"))

        # calibrate it with an unaligned temperature map
        self.unaligned_t_raster_filepaths = glob.glob(
            path.join(self.data_dir, "_T*.tif")
        )

        # calibrate with station measurements
        self.station_t_filepath = path.join(self.data_dir, "station-t.csv")
        self.station_locations_filepath = path.join(
            self.data_dir, "station-locations.csv"
        )
        self.station_t_one_day_filepath = path.join(
            self.data_dir, "station-t-one-day.csv"
        )

        # other parameters
        self.cc_methods = ["factors", "intensity"]
        self.num_steps = 2
        self.num_update_logs = 2
        # self.workspace_dir = path.join(self.data_dir, 'tmp')
        # os.mkdir(self.workspace_dir)

    # def tearDown(self):
    #     shutil.rmtree(self.workspace_dir)

    def test_wrapper_only(self):
        # test that we can use the `UCMWrapper` class without providing any observed
        # temperatures (just to execute the urban cooling model)
        t_refs = 20
        uhi_maxs = 10

        for cc_method in self.cc_methods:
            iuc.UCMWrapper(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths[0],
                t_refs=t_refs,
                uhi_maxs=uhi_maxs,
            ).predict_t_da()

            # test that we can provide an aoi vector
            iuc.UCMWrapper(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths[0],
                aoi_vector_filepath=self.aoi_vector_filepath,
                t_refs=t_refs,
                uhi_maxs=uhi_maxs,
            ).predict_t_da()

        # TODO: test that proper (informative) errors are raised when calling methods
        # that require observed temperatures (e.g., `get_sample_comparison_df`,
        # `get_model_perf_df`...)

    def test_one_day(self):
        ref_et_raster_filepath = self.ref_et_raster_filepaths[0]
        t_raster_filepath = self.t_raster_filepaths[0]
        t_refs = 20
        uhi_maxs = 10

        for cc_method in self.cc_methods:
            # calibrate with map
            # no t_refs/no uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                t_raster_filepaths=t_raster_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # no t_refs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                uhi_maxs=uhi_maxs,
                t_raster_filepaths=t_raster_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # no uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                t_refs=t_refs,
                t_raster_filepaths=t_raster_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # both t_refs/uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                t_refs=t_refs,
                uhi_maxs=uhi_maxs,
                t_raster_filepaths=t_raster_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # unaligned temperature map (no t_refs/no uhi_maxs)
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                t_raster_filepaths=self.unaligned_t_raster_filepaths[0],
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # aoi vector with aligned map
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                aoi_vector_filepath=self.aoi_vector_filepath,
                t_raster_filepaths=self.t_raster_filepaths[0],
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # aoi vector with unaligned map
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                aoi_vector_filepath=self.aoi_vector_filepath,
                t_raster_filepaths=self.unaligned_t_raster_filepaths[0],
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()

            # calibrate with measurements
            # no t_refs/no uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                station_t_filepath=self.station_t_one_day_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # no t_refs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                uhi_maxs=uhi_maxs,
                station_t_filepath=self.station_t_one_day_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # no uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                t_refs=t_refs,
                station_t_filepath=self.station_t_one_day_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # both t_refs/uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                t_refs=t_refs,
                uhi_maxs=uhi_maxs,
                station_t_filepath=self.station_t_one_day_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # aoi vector
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                aoi_vector_filepath=self.aoi_vector_filepath,
                station_t_filepath=self.station_t_one_day_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()

    def test_multiple_days(self):
        t_refs = [20, 21]
        uhi_maxs = [10, 11]

        for cc_method in self.cc_methods:
            # calibrate with map
            # no t_refs/no uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                t_raster_filepaths=self.t_raster_filepaths,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # no t_refs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                uhi_maxs=uhi_maxs,
                t_raster_filepaths=self.t_raster_filepaths,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # no uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                t_refs=t_refs,
                t_raster_filepaths=self.t_raster_filepaths,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # both t_refs/uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                t_refs=t_refs,
                uhi_maxs=uhi_maxs,
                t_raster_filepaths=self.t_raster_filepaths,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # aoi vector
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                aoi_vector_filepath=self.aoi_vector_filepath,
                t_raster_filepaths=self.t_raster_filepaths,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()

            # calibrate with measurements
            # no t_refs/no uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                station_t_filepath=self.station_t_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # no t_refs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                uhi_maxs=uhi_maxs,
                station_t_filepath=self.station_t_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # no uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                t_refs=t_refs,
                station_t_filepath=self.station_t_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # both t_refs/uhi_maxs
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                t_refs=t_refs,
                uhi_maxs=uhi_maxs,
                station_t_filepath=self.station_t_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()
            # aoi vector
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                self.ref_et_raster_filepaths,
                aoi_vector_filepath=self.aoi_vector_filepath,
                station_t_filepath=self.station_t_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).calibrate()

    def test_dates(self):
        cc_method = self.cc_methods[0]
        ref_et_raster_filepath = self.ref_et_raster_filepaths[0]
        t_raster_filepath = self.t_raster_filepaths[0]
        date = "23-07-2020"

        # test the `dates` argument
        # if not providing `station_t_filepath` and not providing the `dates` arg, the
        # `dates` attribute is a numpy array of the length of the ref. evapotransp.
        # raster
        self.assertEqual(
            len(
                iuc.UCMCalibrator(
                    self.lulc_raster_filepath,
                    self.biophysical_table_filepath,
                    cc_method,
                    ref_et_raster_filepath,
                    t_raster_filepaths=t_raster_filepath,
                    num_steps=self.num_steps,
                    num_update_logs=self.num_update_logs,
                ).ucm_wrapper.dates
            ),
            1,  # len(ref_et_raster_filepath)
        )
        # if not providing `station_t_filepath` and providing the `dates` arg, the
        # `dates` attribute is taken from there (although converted to a list of one
        # element only)
        self.assertEqual(
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                t_raster_filepaths=t_raster_filepath,
                dates=date,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).ucm_wrapper.dates,
            [date],
        )
        # if providing `station_t_filepath`, `dates` is taken from there
        self.assertIsNotNone(
            iuc.UCMCalibrator(
                self.lulc_raster_filepath,
                self.biophysical_table_filepath,
                cc_method,
                ref_et_raster_filepath,
                station_t_filepath=self.station_t_one_day_filepath,
                station_locations_filepath=self.station_locations_filepath,
                num_steps=self.num_steps,
                num_update_logs=self.num_update_logs,
            ).ucm_wrapper.dates
        )

    def test_data_array(self):
        cc_method = self.cc_methods[0]
        t_da = iuc.UCMCalibrator(
            self.lulc_raster_filepath,
            self.biophysical_table_filepath,
            cc_method,
            self.ref_et_raster_filepaths,
            station_t_filepath=self.station_t_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps,
            num_update_logs=self.num_update_logs,
        ).predict_t_da()

        # test that the time dimension has a coordinate for each ref. evapotransp.
        # raster filepath
        self.assertEqual(len(t_da["time"]), len(self.ref_et_raster_filepaths))

    def test_data_frames(self):
        cc_method = self.cc_methods[0]
        ucm_calibrator = iuc.UCMCalibrator(
            self.lulc_raster_filepath,
            self.biophysical_table_filepath,
            cc_method,
            self.ref_et_raster_filepaths,
            station_t_filepath=self.station_t_filepath,
            station_locations_filepath=self.station_locations_filepath,
            num_steps=self.num_steps,
            num_update_logs=self.num_update_logs,
        )

        # test that the set of unique sample dates is of the same size as the number of
        # ref. evapotransp. raster filepath
        sample_comparison_df = ucm_calibrator.get_sample_comparison_df()
        self.assertEqual(
            len(sample_comparison_df["date"].unique()),
            len(self.ref_et_raster_filepaths),
        )

        # test that all the columns of the model performance data frame are numeric
        model_perf_df = ucm_calibrator.get_model_perf_df()
        self.assertEqual(
            len(model_perf_df.columns),
            len(
                model_perf_df.select_dtypes(
                    include=["int16", "int32", "int64", "float16", "float32", "float64"]
                ).columns
            ),
        )

        # TODO: test that when providing an AOI vector, the sample comparison df, the
        # number of samples is at most the same that without the AOI vector. I wonder
        # whether this holds in all cases as the AOI can be bigger than the extent
        # implicit in the input rasters/station locations


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.data_dir = "tests/data"

        self.lulc_raster_filepath = path.join(self.data_dir, "lulc.tif")
        self.biophysical_table_filepath = path.join(
            self.data_dir, "biophysical-table.csv"
        )
        self.ref_et_raster_filepaths = glob.glob(
            path.join(self.data_dir, "ref_et*.tif")
        )

        # area of interest vector
        self.aoi_vector_filepath = path.join(self.data_dir, "aoi.gpkg")

        # calibrate with temperature map
        self.t_raster_filepaths = glob.glob(path.join(self.data_dir, "T*.tif"))

        # calibrate it with an unaligned temperature map
        self.unaligned_t_raster_filepaths = glob.glob(
            path.join(self.data_dir, "_T*.tif")
        )

        # calibrate with station measurements
        self.station_t_filepath = path.join(self.data_dir, "station-t.csv")
        self.station_locations_filepath = path.join(
            self.data_dir, "station-locations.csv"
        )
        self.station_t_one_day_filepath = path.join(
            self.data_dir, "station-t-one-day.csv"
        )

        # other parameters
        self.cc_methods = ["factors", "intensity"]
        self.date = "23-07-2020"
        self.num_steps = "2"  # use str instead of int for the CLI
        self.num_update_logs = "2"  # use str instead of int for the CLI
        self.workspace_dir = path.join(self.data_dir, "tmp")
        os.mkdir(self.workspace_dir)

        # TODO: test more possibilities of `args` in `invoke`
        # TODO: test `_dict_from_kws`

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_one_day(self):
        ref_et_raster_filepath = self.ref_et_raster_filepaths[0]
        t_raster_filepath = self.t_raster_filepaths[0]
        unaligned_t_raster_filepath = self.unaligned_t_raster_filepaths[0]
        t_refs = "20"
        uhi_maxs = "10"

        for cc_method in self.cc_methods:
            # calibrate with an aligned map
            result = subprocess.run(
                [
                    "invest-ucm-calibration",
                    self.lulc_raster_filepath,
                    self.biophysical_table_filepath,
                    cc_method,
                    "--ref-et-raster-filepaths",
                    ref_et_raster_filepath,
                    "--t-refs",
                    t_refs,
                    "--uhi-maxs",
                    uhi_maxs,
                    "--t-raster-filepaths",
                    t_raster_filepath,
                    "--num-steps",
                    self.num_steps,
                    "--num-update-logs",
                    self.num_update_logs,
                    "--dst-filepath",
                    path.join(self.workspace_dir, "foo.json"),
                ],
            )
            self.assertEqual(result.returncode, 0)

            # calibrate with an unaligned map
            result = subprocess.run(
                [
                    "invest-ucm-calibration",
                    self.lulc_raster_filepath,
                    self.biophysical_table_filepath,
                    cc_method,
                    "--ref-et-raster-filepaths",
                    ref_et_raster_filepath,
                    "--t-refs",
                    t_refs,
                    "--uhi-maxs",
                    uhi_maxs,
                    "--t-raster-filepaths",
                    unaligned_t_raster_filepath,
                    "--num-steps",
                    self.num_steps,
                    "--num-update-logs",
                    self.num_update_logs,
                    "--dst-filepath",
                    path.join(self.workspace_dir, "foo.json"),
                ]
            )
            self.assertEqual(result.returncode, 0)

            # test the aoi arg
            result = subprocess.run(
                [
                    "invest-ucm-calibration",
                    self.lulc_raster_filepath,
                    self.biophysical_table_filepath,
                    cc_method,
                    "--ref-et-raster-filepaths",
                    ref_et_raster_filepath,
                    "--aoi-vector-filepath",
                    self.aoi_vector_filepath,
                    "--t-raster-filepaths",
                    t_raster_filepath,
                    "--num-steps",
                    self.num_steps,
                    "--num-update-logs",
                    self.num_update_logs,
                    "--dst-filepath",
                    path.join(self.workspace_dir, "foo.json"),
                ]
            )
            self.assertEqual(result.returncode, 0)

            # test the `dates` arg
            result = subprocess.run(
                [
                    "invest-ucm-calibration",
                    self.lulc_raster_filepath,
                    self.biophysical_table_filepath,
                    cc_method,
                    "--ref-et-raster-filepaths",
                    ref_et_raster_filepath,
                    "--t-refs",
                    t_refs,
                    "--uhi-maxs",
                    uhi_maxs,
                    "--t-raster-filepaths",
                    t_raster_filepath,
                    "--dates",
                    self.date,
                    "--num-steps",
                    self.num_steps,
                    "--num-update-logs",
                    self.num_update_logs,
                    "--dst-filepath",
                    path.join(self.workspace_dir, "foo.json"),
                ]
            )
            self.assertEqual(result.returncode, 0)

            # calibrate with measurements
            result = subprocess.run(
                [
                    "invest-ucm-calibration",
                    self.lulc_raster_filepath,
                    self.biophysical_table_filepath,
                    cc_method,
                    "--ref-et-raster-filepaths",
                    ref_et_raster_filepath,
                    "--t-refs",
                    t_refs,
                    "--uhi-maxs",
                    uhi_maxs,
                    "--station-t-filepath",
                    self.station_t_one_day_filepath,
                    "--station-locations-filepath",
                    self.station_locations_filepath,
                    "--num-steps",
                    self.num_steps,
                    "--num-update-logs",
                    self.num_update_logs,
                    "--dst-filepath",
                    path.join(self.workspace_dir, "bar.json"),
                ],
            )
            self.assertEqual(result.returncode, 0)

    def test_multiple_days(self):
        t_refs = [20, 21]
        uhi_maxs = [10, 11]

        for cc_method in self.cc_methods:
            # calibrate with map (test with both space and comma separator)
            for sep in [",", " "]:
                result = subprocess.run(
                    [
                        "invest-ucm-calibration",
                        self.lulc_raster_filepath,
                        self.biophysical_table_filepath,
                        cc_method,
                        "--ref-et-raster-filepaths",
                        _encode_as_cli_arg(self.ref_et_raster_filepaths, sep),
                        "--t-refs",
                        _encode_as_cli_arg(t_refs, sep),
                        "--uhi-maxs",
                        _encode_as_cli_arg(uhi_maxs, sep),
                        "--t-raster-filepaths",
                        _encode_as_cli_arg(self.t_raster_filepaths, sep),
                        "--num-steps",
                        self.num_steps,
                        "--num-update-logs",
                        self.num_update_logs,
                    ]
                )
                self.assertEqual(result.returncode, 0)

            # calibrate with measurements (no need to test again different separators)
            result = subprocess.run(
                [
                    "invest-ucm-calibration",
                    self.lulc_raster_filepath,
                    self.biophysical_table_filepath,
                    cc_method,
                    "--ref-et-raster-filepaths",
                    _encode_as_cli_arg(self.ref_et_raster_filepaths, sep),
                    "--t-refs",
                    _encode_as_cli_arg(t_refs, sep),
                    "--uhi-maxs",
                    _encode_as_cli_arg(uhi_maxs, sep),
                    "--station-t-filepath",
                    self.station_t_filepath,
                    "--station-locations-filepath",
                    self.station_locations_filepath,
                    "--num-steps",
                    self.num_steps,
                    "--num-update-logs",
                    self.num_update_logs,
                ]
            )
            self.assertEqual(result.returncode, 0)

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
