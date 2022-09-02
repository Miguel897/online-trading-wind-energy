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
    # test_start = datetime.datetime.strptime('2022-01-01', '%Y-%m-%d')  # 4 months training
    # test_start = datetime.datetime.strptime('2022-11-01', '%Y-%m-%d')  # 2 months training
    test_start = datetime.datetime.strptime('2022-12-01', '%Y-%m-%d')  # 1 months training
    # test_start = datetime.datetime.strptime('2022-12-31', '%Y-%m-%d')  # 0 months training
    test_end = datetime.datetime.strptime('2023-12-01', '%Y-%m-%d')
    scaling: str = 'nonor'
    global_results_name: str = 'results'
    copy_code: bool = False
    add_ones_column: bool = True
    mode: str = Label.smoothing_mode
    toy_samples: int = 8 * 30 * 24
    solver_method: str = 'ada_grad'
    ada_delta_ro: float = 0.95
    ada_delta_eps: float = 1e-6
    project_q: bool = True
    dynamic_eta: bool = True
    subgradient: bool = True
    q_best_init: bool = False
    q_based_eta: bool = False
    forecast_anchoring: float = None
    feature_space: str = 'original'
    price_offset: int = 1
    augmented_features: tuple = ('DK1_woff_da',)
    augmented_df: int = 3
    eta: float = 1e-3
    eta_0: float = 5e-3
    eta_min: float = 0  # Original
    eta_min_flag: bool = False
    alpha: float = .05
    mu: float = 1.0
    # tolerance: float = 1e-2
    # verbose_steps: int = 1
    verbose_steps: int = 10000
    data_offset: int = 24
    # data_offset: int = 48
    # data_offset: int = 5 * 24
    memory_length: int = 1
    wind_capacity: int = 100
    lambda_value: int = 0
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

# OTHER STUFF

# Labels.met_indexes = tuple(Settings.date_range.strftime('%Y-%m-%d').tolist())
# Labels.met_columns = tuple('C' + str(i) for i in range(Param.n_cases_types + 1))


# Final configuration
Setting.timestamp = get_timestamp_label(underscore=True)
Setting.sim_folder_name = Setting.timestamp + Setting.raw_folder_name
Setting.sim_path = join(Setting.parent_path, Setting.sim_folder_name)
Setting.input_data_path = join(Setting.input_folder_path, Setting.input_data_filename)
Setting.test_interval = (Setting.test_start, Setting.test_end)
Setting.complete_data_set = Setting.test_interval
create_directory(Setting.sim_folder_name, parent_path=Setting.parent_path)

Setting.feature_case = feature_cases[32]
