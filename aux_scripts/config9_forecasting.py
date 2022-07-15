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
    forecasting_mode: str = 'FORECASTING'
    bidding_mode: str = 'BIDDING'
    smoothing_mode: str = 'SMOOTH'
    id: str = 'ID'
    metric_list: tuple = ('MAE', 'MAPE', 'RMSE', 'AOL')
    scaling_types: tuple = ('nonor', 'max', 'mnmx', 'stdr')
    sets: tuple = ('training', 'test')
    # main_wind_features: tuple = ('DK1_won_da', 'DK1_woff_da')
    p_spot: str = 'SpotPriceEUR'
    p_up: str = 'BalancingPowerPriceUpEUR'
    p_dw: str = 'BalancingPowerPriceDownEUR'
    enhanced_forecast: str = 'DK1_FM3_output'
    # dk1real: str = 'DK1_won_real'
    # dk1da: str = 'DK1_won_da'
    # main_wind_features: tuple = ('DK1_won_da',)
    # dk1real: str = 'DK1_woff_real'
    # dk1da: str = 'DK1_woff_da'
    # main_wind_features: tuple = ('DK1_woff_da',)
    # dk1real: str = 'DK2_won_real'
    # dk1da: str = 'DK2_won_da'
    # main_wind_features: tuple = ('DK2_won_da',)
    dk1real: str = 'DK2_woff_real'
    dk1da: str = 'DK2_woff_da'
    main_wind_features: tuple = ('DK2_woff_da',)
    dk1_extra_features: tuple = ('DK1_SchedGen', 'DK1_SolarDahead', 'DK1_TotalLoadDahead')
    neighbor_wind_features: tuple = ('DK2_won_da', 'DK2_woff_da', 'DEATLU_won_da', 'DEATLU_woff_da',
                                     'NO2_won_da', 'SE3_won_da', 'SE4_won_da')
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
    # input_data_filename: str = 'dataset_DK1_wind_entsoe_2015_2021.csv'
    # input_data_filename: str = 'dataset_DK1_wind_entsoe_2015_2021_s_24h.csv'
    input_data_filename: str = 'dataset_DK1_wind_entsoe_2015_2021_scl_1h.csv'
    # test_start = datetime.datetime.strptime('2015-12-26', '%Y-%m-%d')  # Included
    test_start = datetime.datetime.strptime('2015-01-07', '%Y-%m-%d')  # Included
    # test_start = datetime.datetime.strptime('2015-12-31', '%Y-%m-%d')  # Included
    # test_end = datetime.datetime.strptime('2016-01-03', '%Y-%m-%d')  # Not included
    # test_end = datetime.datetime.strptime('2016-02-06', '%Y-%m-%d')
    # test_end = datetime.datetime.strptime('2018-12-31', '%Y-%m-%d')
    test_end = datetime.datetime.strptime('2021-09-30', '%Y-%m-%d')
    # test_end = datetime.datetime.strptime('2016-03-23', '%Y-%m-%d')
    # test_start = datetime.datetime.strptime('2016-11-29', '%Y-%m-%d')  # 1 day before to consider offset
    # test_end = datetime.datetime.strptime('2019-04-23', '%Y-%m-%d')
    scaling: str = 'nonor'  # 'max'
    global_results_name: str = 'results'
    copy_code: bool = False
    add_ones_column: bool = False
    # gen_capacity: float = 100
    feature_space: str = 'original'
    # feature_space: str = 'spline'
    augmented_features: tuple = ('DK1_woff_da',)
    augmented_df: int = 3
    verbose_steps: int = 10000
    # data_offset: int = 24
    data_offset: int = 48
    # data_offset: int = 5 * 24
    memory_length: int = 1
    # memory_length: int = 24
    wind_capacity: int = 3669
    lambda_value: int = 0
    q_j_bounds: tuple = (None, None)
    lagging_features: bool = True
    lagging_values: tuple = (1, 3)  # Lagging hours, Last one not included
    # lagging_values: tuple = (1, 7)  # Lagging hours, Last one not included
    # lagging_values: tuple = (1, 13)  # Lagging hours, Last one not included
    # lagging_values: tuple = (1, 25)  # Lagging hours, Last one not included
    # lagging_values: tuple = (1, 49)  # Lagging hours, Last one not included


Label.binary_features = Label.day_hours + Label.week_days

if Setting.add_ones_column:
    Label.main_wind_features = ('ones',) + Label.main_wind_features
Label.right_feature_order = Label.main_wind_features + Label.dk1_extra_features + \
                            Label.neighbor_wind_features + Label.binary_features
feature_cases = {
    1: Label.main_wind_features,
    2: Label.main_wind_features + Label.dk1_extra_features + Label.binary_features,
    3: Label.main_wind_features + Label.neighbor_wind_features,
    4: Label.main_wind_features + Label.dk1_extra_features + Label.neighbor_wind_features + Label.binary_features,
}
Setting.set_of_regressors = set(reduce(lambda x, y: x + y, feature_cases.values()))

# Final configuration
Setting.timestamp = get_timestamp_label(underscore=True)
Setting.sim_folder_name = Setting.timestamp + Setting.raw_folder_name
Setting.sim_path = join(Setting.parent_path, Setting.sim_folder_name)
Setting.input_data_path = join(Setting.input_folder_path, Setting.input_data_filename)
Setting.test_interval = (Setting.test_start, Setting.test_end)
Setting.complete_data_set = Setting.test_interval
# create_directory(Setting.sim_folder_name, parent_path=Setting.parent_path)

# Setting.alpha = Setting.alpha * Setting.wind_capacity
# Setting.feature_case = feature_cases[1]
Setting.feature_case = feature_cases[1]
