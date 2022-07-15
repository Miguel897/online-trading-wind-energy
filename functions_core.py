# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:11:50 2019

@author: USUARIO
"""
import datetime
import pandas as pd
import numpy as np
from os.path import join
from functions_standard import (
    check_limits,
    append_ones,
    float_dict_to_string as fd2st,
    get_timestamp_label,
    augment_feature_space,
)
from functions_auxiliary import (
    check_q_j,
    compute_metrics,
)
# from optimization import (
#     solve_optimization_model,
#     create_bigadata_nv_model,
# )


def load_data(Label, Setting, add_ones=True, case='real'):
    if case == 'real':
        df = pd.read_csv(Setting.input_data_path, sep=',', decimal='.')
        df.drop(columns=['DK1_won_MAE'], inplace=True)
        df = preprocess_input_data(
            df, Label, Setting, add_ones=add_ones, col_nan_allowed=12.5, tot_nan_allowed=7.5,
        )
        wind, b_data, h_data, features = withdraw_variables(df, Label, Setting, bh_ge_zero=True)
    elif case == 'synthetic':
        wind, b_data, h_data, features = generate_toy_steps_data(Label, Setting)
    elif case == 'convergence':
        wind, features = generate_toy_convergence_data(Label, Setting)
        df = pd.read_csv(Setting.input_data_path, sep=',', decimal='.')
        df = preprocess_input_data(
            df, Label, Setting, add_ones=add_ones, col_nan_allowed=12.5, tot_nan_allowed=7.5,
        )
        _, b_data, h_data, _ = withdraw_variables(df, Label, Setting, bh_ge_zero=True)
    else:
        raise ValueError('Invalid case')

    if Label.price_lag1[0] in Setting.feature_case:
        off = Setting.price_offset
        # add_price_lag(features, b_data, h_data)
        features.loc[features.index[off:], 'psi_p_l1'] = b_data.values[:-off]
        features.loc[features.index[off:], 'psi_m_l1'] = h_data.values[:-off]
        features.loc[features.index[off:], 'tau_l1'] = b_data.values[:-off] / (b_data.values[:-off] + h_data.values[:-off] + 1e-5)

    features = features[list(Setting.feature_case)]

    if Setting.feature_space == 'spline':
        features = augment_feature_space(features, columns=Setting.augmented_features, dgf=Setting.augmented_df)

    return wind, b_data, h_data, features


def load_data_forecast(Label, Setting, add_ones=False):

    df = pd.read_csv(Setting.input_data_path, sep=',', decimal='.')
    df = preprocess_input_data(
        df, Label, Setting, add_ones=add_ones, col_nan_allowed=12.5, tot_nan_allowed=7.5,
    )

    wind_re = df[Label.dk1real].copy()
    wind_da = df[Label.dk1da].copy()
    # df = df[list(Label.right_feature_order)]
    features = df[list(Setting.feature_case)].copy()
    # df.drop(columns=Label.dk1real, inplace=True)
    # features = df.copy()

    # features = features[list(Setting.feature_case)]

    if Setting.lagging_features:
        for off in range(*Setting.lagging_values):
            features.loc[features.index[off:], f'won_re_{off}'] = wind_re.values[:-off]

    if Setting.feature_space == 'spline':
        features = augment_feature_space(features, columns=Setting.augmented_features, dgf=Setting.augmented_df)

    return wind_re, wind_da, features

# def add_price_lag(features, b, h):
#     features['psi_p_l1'] = b.values[1:]


def generate_toy_steps_data(Label, Setting, add_ones=True):

    n_samples = Setting.toy_samples
    n_forth = int(n_samples / 4)
    np.random.seed(seed=17)
    w_forecast = np.random.uniform(10, Setting.wind_capacity - 10, size=n_samples)
    w_true = np.maximum(np.minimum(w_forecast + np.random.normal(0., scale=6., size=n_samples), Setting.wind_capacity), 0)
    mean_diff = np.mean(w_true) - np.mean(w_forecast)
    w_true = np.maximum(np.minimum(w_true - mean_diff / 2, Setting.wind_capacity), 0)
    w_forecast = np.maximum(np.minimum(w_forecast + mean_diff / 2, Setting.wind_capacity), 0)
    mean_diff = np.mean(w_true) - np.mean(w_forecast)
    assert mean_diff <= 0.05

    time_index = pd.date_range(end='2022-12-31 23:00', periods=int(n_samples / 2), freq='H').union(
        pd.date_range(start='2023-01-01 00:00', periods=int(n_samples / 2), freq='H'))
    psi_p = np.concatenate([1 * np.ones(n_forth), 3 * np.ones(n_forth), 1 * np.ones(n_forth), 3 * np.ones(n_forth)])
    psi_m = np.concatenate([3 * np.ones(n_forth), 1 * np.ones(n_forth), 3 * np.ones(n_forth), 1 * np.ones(n_forth)])
    wind = pd.Series(data=w_true, name=Label.dk1real, index=time_index)
    psi_p = pd.Series(data=psi_p, name='psi_p', index=time_index)
    psi_m = pd.Series(data=psi_m, name='psi_m', index=time_index)
    features = pd.DataFrame({Label.ones: np.ones(n_samples), Label.dk1da: w_forecast}, index=time_index)

    mask = (time_index >= Setting.complete_data_set[0]) \
                 & (time_index < Setting.complete_data_set[1])

    wind = wind[mask]
    psi_p = psi_p[mask]
    psi_m = psi_m[mask]
    features = features[mask]

    return wind, psi_p, psi_m, features


def generate_toy_convergence_data(Label, Setting):

    n_samples = Setting.toy_samples
    np.random.seed(seed=17)
    w_forecast = np.random.uniform(10, Setting.wind_capacity - 10, size=n_samples)
    w_true = np.maximum(np.minimum(w_forecast + np.random.normal(0., scale=6., size=n_samples), Setting.wind_capacity), 0)
    mean_diff = np.mean(w_true) - np.mean(w_forecast)
    w_true = np.maximum(np.minimum(w_true - mean_diff / 2, Setting.wind_capacity), 0)
    w_forecast = np.maximum(np.minimum(w_forecast + mean_diff / 2, Setting.wind_capacity), 0)
    mean_diff = np.mean(w_true) - np.mean(w_forecast)
    assert mean_diff <= 0.05

    time_index = pd.date_range(start='2016-12-31 00:00', periods=int(n_samples), freq='H')
    wind = pd.Series(data=w_true, name=Label.dk1real, index=time_index)
    features = pd.DataFrame({Label.ones: np.ones(n_samples), Label.dk1da: w_forecast}, index=time_index)

    mask = (time_index >= Setting.complete_data_set[0]) \
                 & (time_index < Setting.complete_data_set[1])

    wind = wind[mask]
    features = features[mask]

    return wind, features


def preprocess_input_data(
        df, Label, Setting, col_nan_allowed=12.5,
        tot_nan_allowed=7.5, add_ones=False,
):
    # Drop non-required incomplete column
    # if Setting.mode == Label.forecasting_mode:
    #     df.drop(columns=[Label.enhanced_forecast], inplace=True)

    # Filter by time and set an appropriate time index
    df[Label.time] = pd.to_datetime(df[Label.time], format='%d/%m/%Y %H:%M')
    input_mask = (df[Label.time] >= Setting.complete_data_set[0]) \
                 & (df[Label.time] < Setting.complete_data_set[1])
    df = df[input_mask].copy()
    df.set_index(Label.time, drop=True, inplace=True)
    # df.drop(columns=Label.enhanced_forecast, inplace=True)
    # Interpolate data in case of gaps
    nans = df.isnull().sum().sum()
    if nans != 0:
        nan_prtg = nans / df.size * 100
        if nan_prtg > tot_nan_allowed:
            raise ValueError
        calc_nan_dict = dict(zip(list(df.columns.values),
                                 list(df.isnull().sum() / df.shape[0] * 100)))
        calc_nan_dict = {k: v for k, v in calc_nan_dict.items() if v > 0}
        print("Non zero (%) of null values per column: {}".format(fd2st(calc_nan_dict, decimal=2)))
        assert not any([True if v > col_nan_allowed else False for v in calc_nan_dict.values()])

        df.interpolate(inplace=True, limit_direction='both')
        assert df.isnull().sum().sum() == 0, "Unable to interpolate"
        print("Linear interpolation has been performed. Total Nan percentage (%): {:.2f}".format(nan_prtg))
    else:
        print('Df clean without gaps.')
    # Add a column of ones if required
    if add_ones:
        df = append_ones(df)

    return df


def withdraw_variables(df, Label, Setting, bh_ge_zero=False):

    # Withdraw price_mode and wind
    p_im_pos = df[Label.p_dw]  # im_pos associated with producing more than expected
    p_im_neg = df[Label.p_up]
    p_ah = df[Label.p_spot]
    wind_data = df[Label.dk1real]
    b = p_ah - p_im_pos  # psi_+
    h = p_im_neg - p_ah  # psi_-
    if bh_ge_zero:
        b[b < 0] = 0
        h[h < 0] = 0

    # df.drop([Label.real, Label.p_spot, Label.p_up, Label.p_dw], axis=1, inplace=True)
    df = df[list(Label.right_feature_order)].copy()

    if Setting.scaling != 'nonor':
        wind_data = wind_data / Setting.wind_capacity

    return wind_data, b, h, df


def compute_results(X_mat, Y_mat, b, h, q_j_dict, Label, y_bounds=None, prefix=""):

    q_j = np.array(list(q_j_dict.values()))
    prediction = np.array([sum(np.multiply(q_j, X_mat[i, :])) for i in range(len(Y_mat))])
    assert not np.isnan(np.sum(prediction))
    if y_bounds is not None:
        prediction, break_dict = check_limits(prediction, y_bounds)
        break_dict = {prefix + key: value for (key, value) in break_dict.items()}
    assert not np.isnan(np.sum(prediction))  #
    metric_dict = compute_metrics(Y_mat, prediction, b, h, metrics=Label.metric_list)

    if y_bounds is not None:
        return prediction, metric_dict, break_dict
    else:
        return prediction, metric_dict


def compute_fixed_action_q(wind, b_data, h_data, x_data, y_bounds, q_0=None):
    from optimization_lp import (
        create_bigadata_nv_model, standard_solving_configuration,
        solve_optimization_model, solve_optimization_model_direct,
    )

    standard_solving_configuration = {
        'solver_name': 'cplex_direct',
        'solve_options': {'warmstart': True},
        'solver_factory_options': {'options': {
            # 'logfile': 'cplex.log',
            # 'output_clonelog': -1,
            # 'threads': 7
        }},
        'verbose': False,
        'display_info': False,
        'write_info': False,
        'solver_file_name': 'solver_info',
        'model_file_name': 'pyomo_model',
        'file_extension': '.txt',
        'saving_path': '',
        'timestamp': '',
    }

    standard_solving_configuration['solver_factory_options']['options']['simplex_tolerances_optimality'] = 1e-2
    standard_solving_configuration['solver_factory_options']['options']['lpmethod'] = 2  # Dual simplex

    data = {
        'y_i': wind,
        'x_ij': x_data,
        'b_i': b_data,
        'h_i': h_data,
        'y_bounds': y_bounds,
        'q_j_bounds': (None, None),
    }
    if q_0 is not None:
        data['q_0'] = q_0

    model = create_bigadata_nv_model(data)
    # solved_model, solver_status, solver_additional_information = solve_optimization_model_direct(
    solved_model, solver_status, solver_additional_information = solve_optimization_model_direct(
        model, standard_solving_configuration)
    q_j = list(solved_model.q_j[j].value for j in solved_model.j)
    obj_func = solver_additional_information['upper_bound']

    # from main1_gradient_solver import newton_solver
    # q_n0 = pd.DataFrame({c: np.array([1e-2]) for c in x_data.columns})
    # q_n0['DK1_won_da'] = 1
    # q_n, obj_n = newton_solver(q_n0, data)
    # q_n = list(q_n.values)
    # q_j, obj_func = q_n, obj_n

    return q_j, obj_func


def compute_feasible_set_diameter(wind, x_data, y_bounds, q_0=None):
    from optimization_lp import (
        create_feasible_set_diameter_model, solve_optimization_model_direct,
    )

    standard_solving_configuration = {
        'solver_name': 'cplex_direct',
        'solve_options': {},
        'solver_factory_options': {'options': {
            # 'logfile': 'cplex.log',
            # 'output_clonelog': -1,
            # 'threads': 7
            'optimalitytarget': 3,
        }},
        'verbose': True,
        # 'display_info': True,
        'display_info': False,
        'write_info': True,
        'solver_file_name': 'solver_info',
        'model_file_name': 'pyomo_model',
        'file_extension': '.txt',
        'saving_path': '',
        'timestamp': '',
    }

    standard_solving_configuration['solver_factory_options']['options']['simplex_tolerances_optimality'] = 1e-2
    standard_solving_configuration['solver_factory_options']['options']['lpmethod'] = 2  # Dual simplex

    data = {
        # 'y_i': wind[100:2000],
        # 'y_i': wind[100:200],
        'y_i': wind[100:250],
        # 'y_i': wind[200:300],
        # 'x_ij': x_data[100:2000],
        # 'x_ij': x_data[100:200],
        'x_ij': x_data[100:250],
        # 'x_ij': x_data[200:300],
        'y_bounds': y_bounds,
        'q_j_bounds': (None, None),
    }
    # x_ij, y_i = data['x_ij'][100:2000], data['y_i'][100:2000]

    if q_0 is not None:
        data['q_0'] = q_0

    model = create_feasible_set_diameter_model(data)
    solved_model, solver_status, solver_additional_information = solve_optimization_model_direct(
        model, standard_solving_configuration)
    q1 = list(solved_model.q1[j].value for j in solved_model.j)
    q2 = list(solved_model.q2[j].value for j in solved_model.j)
    print(q1)
    print(q2)
    from scipy.spatial.distance import euclidean
    distance = euclidean(q1, q2)
    print(distance)
    pass
    # obj_func = solver_additional_information['upper_bound']

    # return q_j, obj_func

# def compute_coefficients(
#         data, labels, optimization_configuration
# ):
#
#     model = create_bigadata_nv_model(data)
#     model, solver_status, solver_additional_information = solve_optimization_model(
#         model, optimization_configuration
#     )
#     q_j = [model.q_j[j].value for j in model.j]
#     q_j_dict, err = check_q_j(q_j, labels)
#     solver_status['coef_error_rate'] = err
#
#     return q_j_dict, solver_status
#
#
# def save_results(
#         metric_container, q_j_container,
#         benchmark_container, prediction_container,
#         Label, Param, Setting,
# ):
#     f_name = get_timestamp_label(underscore=True) + Setting.global_results_name
#     writer = pd.ExcelWriter(
#         join(Setting.simulation_full_path, f_name + Label.excel), engine='xlsxwriter'
#     )
#
#     # Save metrics
#     for metric in Label.metric_list:
#         for data_set, prefix in zip(Label.sets, ['in_', 'out_']):
#             pd.DataFrame(
#                 metric_container[data_set][metric],
#                 index=Label.met_indexes, columns=Label.met_columns
#             ).to_excel(writer, sheet_name=prefix + metric)
#
#     # Save q_j containers
#     q_j_container.to_excel(writer, sheet_name='q_j_values')
#
#     # Save predictions
#     prediction_index = tuple(pd.date_range(
#         start=Setting.test_set[0],
#         end=Setting.test_set[1] - datetime.timedelta(hours=1),
#         freq='1H').strftime('%Y-%m-%d_%H:%M:%S').tolist())
#
#     n_pred = len(benchmark_container['real'])
#     prd_lst = [
#         np.reshape(np.array(sum([[d] * Param.dayh for d in range(int(n_pred / Param.dayh))], [])), (-1, 1)),
#         benchmark_container['real'].reshape((-1, 1)),
#         np.reshape(benchmark_container['bench'], (-1, 1)),
#     ]
#
#     for case in Setting.cases:
#         prd_lst.append(np.reshape(prediction_container[case], (-1, 1)))
#
#     pd.DataFrame(
#         np.concatenate(prd_lst, axis=1),
#         index=prediction_index,
#         columns=['day', 'real', 'bench'] + ['C{}'.format(c) for c in Setting.cases]
#     ).to_excel(writer, sheet_name='Predictions')
#
#     # Write all sheets to file
#     writer.save()

