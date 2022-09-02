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
    raw_folder_name: str = 'lp_nv_toy'
    input_data_filename: str = 'data_DK1_wind_entsoe_2015_2021_scl_1h.csv'
    # test_start = datetime.datetime.strptime('2015-12-26', '%Y-%m-%d')  # Included
    # test_start = datetime.datetime.strptime('2015-12-30', '%Y-%m-%d')  # Included
    # test_start = datetime.datetime.strptime('2015-07-01', '%Y-%m-%d')  # Included
    test_initial = datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')  # Offset day
    test_end = datetime.datetime.strptime('2023-12-01', '%Y-%m-%d')
    # test_end = datetime.datetime.strptime('2023-01-04', '%Y-%m-%d')
    # test_initial = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')  # Included
    scaling: str = 'nonor'  # 'max'
    global_results_name: str = 'results'
    copy_code: bool = False
    add_ones_column: bool = True
    # mode: str = Label.forecasting_mode
    # price_mode: str = Label.smoothing_mode
    price_mode: str = Label.bidding_mode
    # n_samples: int = 100
    toy_samples: int = 8 * 30 * 24
    # gen_capacity: float = 100
    feature_space: str = 'original'
    # feature_space: str = 'spline'
    # price_offset: int = 1
    # price_offset: int = 6
    price_offset: int = 1
    single_feature: bool = False
    augmented_features: tuple = ('DK1_woff_da',)
    augmented_df: int = 3
    mu: float = 1
    verbose_steps: int = 10000
    data_offset: int = 24 * 0
    chunk_length: int = 24 * 1
    up_step: int = 12
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
    21: (Label.dk1da,),
    31: (Label.ones, Label.dk1da),
    32: (Label.dk1da,),
}
Setting.set_of_regressors = set(reduce(lambda x, y: x + y, feature_cases.values()))

# Final configuration
Setting.timestamp = get_timestamp_label(underscore=True)
Setting.sim_folder_name = Setting.timestamp + Setting.raw_folder_name
Setting.sim_path = join(Setting.parent_path, Setting.sim_folder_name)
Setting.input_data_path = join(Setting.input_folder_path, Setting.input_data_filename)
Setting.test_start = Setting.test_initial - datetime.timedelta(hours=Setting.data_offset)
Setting.test_interval = (Setting.test_start, Setting.test_end)
Setting.complete_data_set = Setting.test_interval
create_directory(Setting.sim_folder_name, parent_path=Setting.parent_path)

Setting.feature_case = feature_cases[32]
