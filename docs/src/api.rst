=====================
Reference Guide (API)
=====================

The InVEST urban cooling model calibration library provides two classes:

The `UCMWrapper` class is intended to provide a Pythonic and object-oriented interface to the urban cooling model which can be used to interactively experiment with the model (e.g., trying the results obtained with different parameters), and includes a set of additional useful methods.

.. autoclass:: invest_ucm_calibration.UCMWrapper
   :members: __init__, predict_t, predict_t_da, get_sample_comparison_df, get_model_perf_df

The `UCMCalibrator` class inherits the `Annealer` from the `simanneal <https://github.com/perrygeo/simanneal>`_ package and makes use of the `UCMWrapper` class in order to apply the simulated annealing procedure to the InVEST urban cooling model.

.. autoclass:: invest_ucm_calibration.UCMCalibrator
   :members: __init__, calibrate, predict_t, predict_t_da, get_sample_comparison_df, get_model_perf_df
