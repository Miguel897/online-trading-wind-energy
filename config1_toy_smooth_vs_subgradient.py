import datetime
from dataclasses import dataclass
from os import getcwd
from os.path import join
from functools import reduce
from config_label import Label
from functions_standard import (
    get_timestamp_label,
    create_directory,
)

try:
    from private_files.functions_private import update_context_variables
    input_path, simulation_path, _ = update_context_variables()
except ModuleNotFoundError:
    input_path, simulation_path = getcwd(), getcwd()


@dataclass
class Setting:
    set_of_regressors: tuple
    in_data_path: str
    sim_folder_path: str
    complete_data_set: str
    training_days: int
    test_len: int
    n_cases: int
    wind_limits: tuple
    sim_path: str
    input_data_path: str
    sim_folder_name: str
    timestamp: str
    test_interval: tuple
    feature_case: tuple
    input_folder_path: str = input_path
    parent_path: str = simulation_path
    raw_folder_name: str = 'online_toy'
    input_data_filename: str = 'data_DK1_wind_entsoe_2015_2021_scl_1h.csv'
    test_start = datetime.datetime.strptime('2016-12-31', '%Y-%m-%d')  # The whole day is loaded to get the penalty lags
    test_end = datetime.datetime.strptime('2018-01-01', '%Y-%m-%d')  # This date not included in the interval
    scaling: str = 'nonor'
    global_results_name: str = 'results'
    copy_code: bool = False
    add_ones_column: bool = True
    mode: str = Label.bidding_mode
    # Gradient options
    solver_method: str = 'ada_grad'
    # Reference: [1] Zeiler, Matthew D. "Adadelta: an adaptive learning rate method." arXiv preprint arXiv:1212.5701 (2012).
    ada_delta_ro: float = 0.95  # Parameters as in [1]
    ada_delta_eps: float = 1e-6  # Parameters as in [1]
    project_q: bool = True
    dynamic_eta: bool = True
    subgradient: bool = False  # <---- CHANGE THIS to True to use subgradients and False to use the smooth approximation
    q_best_init: bool = False
    q_based_eta: bool = False
    forecast_anchoring: float = None
    feature_space: str = 'original'
    price_offset: int = 1
    augmented_features: tuple = ('DK1_woff_da',)
    augmented_df: int = 3
    eta: float = 5e-3
    eta_0: float = 5e-3
    eta_min: float = 0
    alpha: float = 20  # <---- Smoothing parameter. In the paper is used 0.05, 5 and 20.
    mu: float = 1.0
    verbose_steps: int = 10000
    data_offset: int = 24  # Number of hours at the start of the dataset used for auxiliary operations
    memory_length: int = 1  # Number of points to compute the update
    wind_capacity: int = 100  # In MWs
    lambda_value: int = 0  # Regularization if any
    q_j_bounds: tuple = (None, None)

# FEATURES

Label.binary_features = Label.day_hours + Label.week_days
if Setting.add_ones_column:
    Label.main_wind_features = ('ones',) + Label.main_wind_features
    Label.dk_wind_features = ('ones',) + Label.dk_wind_features
Label.right_feature_order = Label.dk_wind_features
# Label.right_feature_order = Label.main_wind_features + Label.dk1_extra_features +\
#                     Label.neighbor_wind_features + Label.binary_features
feature_cases = {
    1: Label.main_wind_features,
    2: Label.main_wind_features + Label.dk1_extra_features + Label.binary_features,
    3: Label.main_wind_features + Label.neighbor_wind_features,
    4: Label.main_wind_features + Label.dk1_extra_features + Label.neighbor_wind_features + Label.binary_features,
    5: Label.dk_wind_features,
    11: Label.main_wind_features + Label.price_lag1,
    15: Label.dk_wind_features + Label.price_lag1,
    31: (Label.ones, Label.dk1da),
    32: (Label.dk1da,),
}
Setting.set_of_regressors = set(reduce(lambda x, y: x + y, feature_cases.values()))

# Final configuration
Setting.timestamp = get_timestamp_label(underscore=True)
Setting.sim_folder_name = Setting.timestamp + Setting.raw_folder_name
Setting.sim_path = join(Setting.parent_path, Setting.sim_folder_name)
Setting.input_data_path = join(Setting.input_folder_path, Setting.input_data_filename)
Setting.test_interval = (Setting.test_start, Setting.test_end)
Setting.complete_data_set = Setting.test_interval
Setting.feature_case = feature_cases[32]
create_directory(Setting.sim_folder_name, parent_path=Setting.parent_path)
