# DEFAULT_INITIAL_SOLUTION = [500, 100, 0.6, 0.2, 0.2]
DEFAULT_UCM_PARAMS = {
    't_air_average_radius': 500,
    'green_area_cooling_distance': 100,
    'cc_weight_shade': 0.6,
    'cc_weight_albedo': 0.2,
    'cc_weight_eti': 0.2
}
DEFAULT_EXTRA_UCM_ARGS = {'do_valuation': False}

DEFAULT_METRIC = 'R2'
DEFAULT_STEPSIZE = 0.3
DEFAULT_NUM_STEPS = 100
DEFAULT_NUM_UPDATE_LOGS = 100

DEFAULT_MODEL_PERF_NUM_RUNS = 10

MIN_KERNEL_DIST_EPS = 0.1
