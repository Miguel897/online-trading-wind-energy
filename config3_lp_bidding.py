
import os
import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from itertools import product
from os import getcwd
from os.path import join
from functools import reduce
from functions_standard import (
    get_timestamp_label,
    PlatformContext,
    SimulationContext,
    create_directory,
    save_code2,
)

platform_context = PlatformContext()
simulation_list = ['e1']

if platform_context.is_pc:
    input_path = r'C:\DATOS\01_CON_RESPALDO\Input\20210929_online'
    simulation_path = r'C:\DATOS\01_CON_RESPALDO\Simulaciones'
    # copy_source_files(folder_timestamp)
    timestamp = get_timestamp_label(underscore=True)
    # simulation_path = join(simulation_path, timestamp + 'online_nv')
    # save_code2(simulation_path=simulation_path, ignore=('z_*', 'test_*', '*.xlsx', '*.csv'))
    platform_context_parameters = {
        'threads': 7,
    }
    simulation_context_parameters = {
        'full_simulation_list': simulation_list,
        'default_simulation_mode': 'all',
        # 'default_simulation_mode': 'index',  # To run a single case in PC ... \
        # 'sim_cases_indexes': [5],  # ... uncomment this two lines
    }
elif platform_context.is_ada:
    input_path = join(getcwd(), 'input')
    simulation_path = join(getcwd(), 'Simulaciones')
    platform_context_parameters = {
        'threads': 7,
        'temporary_dir': simulation_path,
    }
    simulation_context_parameters = {
        'default_simulation_mode': 'all',
        'full_simulation_list': simulation_list,
    }
elif platform_context.is_picasso:
    input_path = join(getcwd(), 'input')
    simulation_path = join(getcwd(), 'Simulaciones')
    platform_context_parameters = {
        'threads': 4,
        'temporary_dir': simulation_path,
    }
    simulation_context_parameters = {
        'default_simulation_mode': 'batch',
        'full_simulation_list': simulation_list,
        # 'batch_size': 1,  # Number of cases to run per job. Required mode: batch
    }
else:
    raise ValueError('Unable to detect platform.')

platform_context.update_environ_variables(**platform_context_parameters)
script_simulation_list = SimulationContext(**simulation_context_parameters).script_simulation_list
script_simulation_dict = {i: i for i in script_simulation_list}


@dataclass
class Label:
    met_indexes: tuple
    met_columns: tuple
    binary_features: tuple
    right_feature_order: tuple
    excel: str = '.xlsx'
    ones: str = 'ones'
    time: str = 'TimeUTC'
    dk1real: str = 'DK1_won_real'
    dk1da: str = 'DK1_won_da'
    dk1mae: str = 'DK1_won_MAE'
    p_spot: str = 'SpotPriceEUR'
    p_up: str = 'BalancingPowerPriceUpEUR'
    p_dw: str = 'BalancingPowerPriceDownEUR'
    enhanced_forecast: str = 'DK1_FM3_output'
    forecasting_mode: str = 'FORECASTING'
    bidding_mode: str = 'BIDDING'
    smoothing_mode: str = 'SMOOTH'
    id: str = 'ID'
    metric_list: tuple = ('MAE', 'MAPE', 'RMSE', 'AOL')
    scaling_types: tuple = ('nonor', 'max', 'mnmx', 'stdr')
    sets: tuple = ('training', 'test')
    main_wind_features: tuple = ('DK1_won_da', 'DK1_woff_da')
    dk1_extra_features: tuple = ('DK1_SchedGen', 'DK1_SolarDahead', 'DK1_TotalLoadDahead')
    neighbor_wind_features: tuple = ('DK2_won_da', 'DK2_woff_da', 'DEATLU_won_da', 'DEATLU_woff_da',
                                     'NO2_won_da', 'SE3_won_da', 'SE4_won_da')
    dk_wind_features: tuple = ('DK1_won_da', 'DK1_woff_da', 'DK2_won_da', 'DK2_woff_da')
    day_hours: tuple = tuple('hour' + str(i) for i in range(24))
    week_days: tuple = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
    price_lag1: tuple = ('psi_p_l1', 'psi_m_l1', 'tau_l1')


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
    raw_folder_name: str = 'online_nv'
    input_data_filename: str = 'dataset_DK1_wind_entsoe_2015_2021_scl_1h.csv'
    # test_start = datetime.datetime.strptime('2015-12-26', '%Y-%m-%d')  # Included
    # test_start = datetime.datetime.strptime('2015-12-30', '%Y-%m-%d')  # Included
    # test_start = datetime.datetime.strptime('2015-07-01', '%Y-%m-%d')  # Included
    # test_initial = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')  # Included
    test_initial = datetime.datetime.strptime('2016-07-01', '%Y-%m-%d')  # Included
    # test_end = datetime.datetime.strptime('2016-07-03', '%Y-%m-%d')  # Included
    # test_end = datetime.datetime.strptime('2016-01-03', '%Y-%m-%d')  # Not included
    test_end = datetime.datetime.strptime('2022-01-01', '%Y-%m-%d')  # Not included
    # test_end = datetime.datetime.strptime('2022-01-01', '%Y-%m-%d')  # Not included
    # test_end = datetime.datetime.strptime('2016-02-03', '%Y-%m-%d')  # Not included
    # test_end = datetime.datetime.strptime('2017-01-01', '%Y-%m-%d')  # Not included
    # test_end = datetime.datetime.strptime('2016-02-06', '%Y-%m-%d')
    # test_end = datetime.datetime.strptime('2018-12-31', '%Y-%m-%d')
    # test_end = datetime.datetime.strptime('2016-03-23', '%Y-%m-%d')
    # test_start = datetime.datetime.strptime('2016-11-29', '%Y-%m-%d')  # 1 day before to consider offset
    # test_end = datetime.datetime.strptime('2019-04-23', '%Y-%m-%d')
    scaling: str = 'nonor'  # 'max'
    global_results_name: str = 'results'
    copy_code: bool = False
    add_ones_column: bool = True
    # mode: str = Label.forecasting_mode
    # price_mode: str = Label.smoothing_mode
    price_mode: str = Label.bidding_mode
    # price_mode: str = Label.forecasting_mode
    # n_samples: int = 100
    # gen_capacity: float = 100
    feature_space: str = 'original'
    # feature_space: str = 'spline'
    # price_offset: int = 1
    # price_offset: int = 6
    price_offset: int = 1
    single_feature: bool = False
    augmented_features: tuple = ('DK1_woff_da',)
    augmented_df: int = 3
    # solver_method: str = 'steepest'
    # iteration_limit: int = int(5 * 1e5)
    # iteration_limit: int = int(1 * 1e3)
    mu: float = 1
    verbose_steps: int = 10000
    # data_offset: int = 48
    # data_offset: int = 24 * (30 + 1) * 6
    # data_offset: int = 24
    data_offset: int = 24 * 30 * 6
    # data_offset: int = 24 * 30
    # chunk_length: int = 24
    chunk_length: int = 24 * 30 * 6
    # chunk_length: int = 24 * 30
    up_step: int = 24
    # up_step: int = 12
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
Label.right_feature_order = Label.dk_wind_features + (Label.dk1mae,)
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
    22: (Label.dk1mae,),
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
Setting.test_start = Setting.test_initial - datetime.timedelta(hours=Setting.data_offset)
Setting.test_interval = (Setting.test_start, Setting.test_end)
Setting.complete_data_set = Setting.test_interval
create_directory(Setting.sim_folder_name, parent_path=Setting.parent_path)

# Setting.alpha = Setting.alpha * Setting.wind_capacity
# Setting.feature_case = feature_cases[1]
# Setting.feature_case = feature_cases[5]
Setting.feature_case = feature_cases[15]
# Setting.feature_case = feature_cases[21]
# Setting.feature_case = feature_cases[22]  # Second step <----
