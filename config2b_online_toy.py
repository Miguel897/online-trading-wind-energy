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
    # test_start = datetime.datetime.strptime('2015-12-26', '%Y-%m-%d')  # Included
    # test_start = datetime.datetime.strptime('2015-12-30', '%Y-%m-%d')  # Included
    # test_start = datetime.datetime.strptime('2015-06-30', '%Y-%m-%d')  # Included
    # Crossval
    # test_start = datetime.datetime.strptime('2015-07-01', '%Y-%m-%d')  # Included
    # test_end = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')
    # Test
    # test_start = datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')
    # test_start = datetime.datetime.strptime('2022-01-01', '%Y-%m-%d')  # 4 months training
    # test_start = datetime.datetime.strptime('2022-11-01', '%Y-%m-%d')  # 2 months training
    test_start = datetime.datetime.strptime('2022-12-01', '%Y-%m-%d')  # 1 months training
    # test_start = datetime.datetime.strptime('2022-12-31', '%Y-%m-%d')  # 0 months training
    test_end = datetime.datetime.strptime('2023-12-01', '%Y-%m-%d')
    #
    # test_start = datetime.datetime.strptime('2015-12-31', '%Y-%m-%d')  # Included
    # test_end = datetime.datetime.strptime('2016-01-03', '%Y-%m-%d')  # Not included
    # test_end = datetime.datetime.strptime('2016-02-06', '%Y-%m-%d')
    # test_end = datetime.datetime.strptime('2016-03-23', '%Y-%m-%d')
    # test_start = datetime.datetime.strptime('2016-11-29', '%Y-%m-%d')  # 1 day before to consider offset
    # test_end = datetime.datetime.strptime('2019-04-23', '%Y-%m-%d')
    scaling: str = 'nonor'  # 'max'
    global_results_name: str = 'results'
    copy_code: bool = False
    add_ones_column: bool = True
    # mode: str = Label.forecasting_mode
    mode: str = Label.smoothing_mode
    # mode: str = Label.bidding_mode
    # Gradient options
    # n_samples: int = 100
    # toy_samples: int = 100
    toy_samples: int = 8 * 30 * 24
    # gen_capacity: float = 100
    # solver_method: str = 'newton'
    # solver_method: str = 'steepest'
    solver_method: str = 'ada_grad'
    # ada_delta_ro: float = 0.99
    ada_delta_ro: float = 0.95
    # ada_delta_ro: float = 0.99
    ada_delta_eps: float = 1e-6
    # project_q: bool = False
    # dynamic_eta: bool = False
    # subgradient: bool = False
    project_q: bool = True
    dynamic_eta: bool = True
    subgradient: bool = True
    q_best_init: bool = False
    q_based_eta: bool = False
    forecast_anchoring: float = None
    feature_space: str = 'original'
    # feature_space: str = 'spline'
    price_offset: int = 1
    # price_offset: int = 6
    # price_offset: int = 24
    augmented_features: tuple = ('DK1_woff_da',)
    augmented_df: int = 3
    # solver_method: str = 'steepest'
    # iteration_limit: int = int(5 * 1e5)
    # iteration_limit: int = int(1 * 1e3)
    # eta: float = 1e-10
    # eta: float = 1 * 1e-4
    # eta: float = .1 * 1e-4
    eta: float = 1e-3
    # eta_0: float = 1e-3
    eta_0: float = 5e-3
    eta_min: float = 0  # Original
    eta_min_flag: bool = False
    alpha: float = .05
    # mu: float = .3
    mu: float = 1.0
    # mu: float = 0.8
    # mu: float = 0.7
    # mu: float = 0.5
    # alpha: float = .25
    # alpha: float = 0.02
    # tolerance: float = 1e-2
    # verbose_steps: int = 1
    verbose_steps: int = 10000
    data_offset: int = 24
    # data_offset: int = 48
    # data_offset: int = 5 * 24
    memory_length: int = 1
    # memory_length: int = 24
    # memory_length: int = 48
    # memory_length: int = 2 * 24
    # memory_length: int = 5 * 24
    # Other options
    # n_cases_types: int = 5
    # dayh: int = 24
    # training_months: int = 6
    # month_len: int = 30
    # gap: int = 1  # Days between training and test
    # test_days: int = 1
    wind_capacity: int = 100
    lambda_value: int = 0
    q_j_bounds: tuple = (None, None)

# TIME
# Paper dates
# Settings.test_start = datetime.datetime.strptime('2016-06-03', '%Y-%m-%d')
# Settings.test_end = datetime.datetime.strptime('2016-06-05', '%Y-%m-%d')  # Last day not included
# Settings.test_start = datetime.datetime.strptime('2016-06-03', '%Y-%m-%d')
# Settings.test_end = datetime.datetime.strptime('2016-11-30', '%Y-%m-%d')
# Settings.test_start = datetime.datetime.strptime('2015-08-07', '%Y-%m-%d')
# Settings.test_end = datetime.datetime.strptime('2015-08-09', '%Y-%m-%d')  # Last day not included
# Forecasting validation set
# Settings.test_start = datetime.datetime.strptime('2015-08-07', '%Y-%m-%d')
# Settings.test_end  = datetime.datetime.strptime('2016-02-03', '%Y-%m-%d')
# Forecasting results set
# Settings.test_start = datetime.datetime.strptime('2016-02-04', '%Y-%m-%d')
# Settings.test_end = datetime.datetime.strptime('2019-04-23', '%Y-%m-%d')
#
#
# Param.wind_limits = (0, Param.wind_capacity)  # (0, 1)
# Param.training_days = Param.training_months * Param.month_len
# Param.n_cases = len(Settings.cases)
# Settings.test_set = (Settings.test_start, Settings.test_end)
# assert (Settings.test_end - Settings.test_start).seconds == 0
# Settings.start_date = Settings.test_set[0] - datetime.timedelta(
#     days=(Param.training_days + Param.gap + 1))
# Settings.complete_data_set = [Settings.start_date, Settings.test_set[1]]
# Settings.date_range = pd.date_range(
#     start=Settings.test_set[0].date(),
#     end=Settings.test_set[1].date() - datetime.timedelta(days=1),  # Last day not included
# )
# Param.test_len = len(Settings.date_range)


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

# Setting.alpha = Setting.alpha * Setting.wind_capacity
# Setting.feature_case = feature_cases[1]
# Setting.feature_case = feature_cases[5]
# Setting.feature_case = feature_cases[15]
# Setting.feature_case = feature_cases[31]
Setting.feature_case = feature_cases[32]
