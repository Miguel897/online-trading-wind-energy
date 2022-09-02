import pandas as pd
import numpy as np
from functions_standard import (
    check_limits,
    append_ones,
    float_dict_to_string as fd2st,
    augment_feature_space,
    compute_metrics,
)
from optimization_lp import (
    create_bigadata_nv_model,
    standard_solving_configuration,
)
try:
    from additional_files.optimization_utils import solve_optimization_model_direct as solve_optimization_model
except ModuleNotFoundError:
    from optimization_lp import solve_optimization_model


def load_data(Label, Setting, add_ones=True, case='case_study'):
    if case == 'case_study':
        df = pd.read_csv(Setting.input_data_path, sep=',', decimal='.')
        df.drop(columns=[Label.price_lag1], inplace=True)
        df = preprocess_input_data(df, Label, Setting, add_ones=add_ones)
        wind, psi_p, psi_m, features = withdraw_variables(df, Label, Setting, bh_ge_zero=True)
    elif case == 'dynamic':
        wind, psi_p, psi_m, features = generate_toy_dynamic_data(Label, Setting)
    elif case == 'smooth':
        wind, features = generate_toy_smooth_vs_subgradient_data(Label, Setting)
        df = pd.read_csv(Setting.input_data_path, sep=',', decimal='.')
        df = preprocess_input_data(df, Label, Setting, add_ones=add_ones)
        _, psi_p, psi_m, _ = withdraw_variables(df, Label, Setting, bh_ge_zero=True)
    else:
        raise ValueError('Invalid case')

    if Label.price_lag1[0] in Setting.feature_case:
        off = Setting.price_offset
        features.loc[features.index[off:], 'psi_p_l1'] = psi_p.values[:-off]
        features.loc[features.index[off:], 'psi_m_l1'] = psi_m.values[:-off]
        features.loc[features.index[off:], 'tau_l1'] = psi_p.values[:-off] \
                                                       / (psi_p.values[:-off] + psi_m.values[:-off] + 1e-5)

    features = features[list(Setting.feature_case)]

    if Setting.feature_space == 'spline':
        features = augment_feature_space(features, columns=Setting.augmented_features, dgf=Setting.augmented_df)

    return wind, psi_p, psi_m, features


def load_data_forecast(Label, Setting, csv_format=',', add_ones=False):
    """Not used in the main simulations.
    """

    if csv_format == ',':
        sep, decimal = ',', '.'
    elif csv_format == ';':
        sep, decimal = ';', ','
    else:
        raise ValueError('Invalid csv option.')

    df = pd.read_csv(Setting.input_data_path, sep=sep, decimal=decimal)
    df = preprocess_input_data(df, Label, Setting, add_ones=add_ones)

    wind_re = df[Label.dk1real].copy()
    wind_da = df[Label.dk1da].copy()
    features = df[list(Setting.feature_case)].copy()

    if Setting.lagging_features:
        for off in range(*Setting.lagging_values):
            features.loc[features.index[off:], f'won_re_{off}'] = wind_re.values[:-off]

    if Setting.feature_space == 'spline':
        features = augment_feature_space(features, columns=Setting.augmented_features, dgf=Setting.augmented_df)

    return wind_re, wind_da, features


def generate_toy_dynamic_data(Label, Setting):
    """Generates the wind and penalty data for the toy example that that compares OLNV against LP.
    """

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


def generate_toy_smooth_vs_subgradient_data(Label, Setting):
    """Generates the wind data for the toy example that compares the smooth and subgradient
    implementations of OLNV.
    """

    # Only the left side of the interval is included
    time_index = pd.date_range(start=Setting.complete_data_set[0],
                               end=Setting.complete_data_set[1], freq='H', closed='left')
    n_samples = len(time_index)
    np.random.seed(seed=17)

    # Generates two samples with almost equal mean
    w_forecast = np.random.uniform(10, Setting.wind_capacity - 10, size=n_samples)
    w_true = np.maximum(np.minimum(w_forecast + np.random.normal(0., scale=6., size=n_samples), Setting.wind_capacity), 0)
    mean_diff = np.mean(w_true) - np.mean(w_forecast)
    w_true = np.maximum(np.minimum(w_true - mean_diff / 2, Setting.wind_capacity), 0)
    w_forecast = np.maximum(np.minimum(w_forecast + mean_diff / 2, Setting.wind_capacity), 0)
    mean_diff = np.mean(w_true) - np.mean(w_forecast)
    assert mean_diff <= 0.05

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
    p_im_pos = df[Label.p_dw]  # price related to producing more than expected
    p_im_neg = df[Label.p_up]
    p_ah = df[Label.p_spot]
    wind_data = df[Label.dk1real]
    psi_p = p_ah - p_im_pos  # psi_+
    psi_n = p_im_neg - p_ah  # psi_-
    if bh_ge_zero:
        psi_p[psi_p < 0] = 0
        psi_n[psi_n < 0] = 0

    # df.drop([Label.real, Label.p_spot, Label.p_up, Label.p_dw], axis=1, inplace=True)
    df = df[list(Label.right_feature_order)].copy()

    if Setting.scaling != 'nonor':
        wind_data = wind_data / Setting.wind_capacity

    return wind_data, psi_p, psi_n, df


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


def compute_optimal_lp_q(wind, psi_p, psi_m, x_data, y_bounds, q_0=None, config=None, extra_options=None):

    if config is None:
        config = standard_solving_configuration.copy()
    if extra_options is not None:
        config['solver_factory_options']['options'].update(extra_options)

    data = {
        'y_i': wind,
        'x_ij': x_data,
        'psi_p_i': psi_p,
        'psi_m_i': psi_m,
        'lamb': 0,
        'y_bounds': y_bounds,
        'q_j_bounds': (None, None),
    }
    if q_0 is not None:
        data['q_0'] = q_0
        config['solve_options']['warmstart'] = True

    model = create_bigadata_nv_model(data)
    solved_model, solver_status, solver_additional_information = solve_optimization_model(model, config)
    q_j = list(solved_model.q_j[j].value for j in solved_model.j)
    obj_func = solver_additional_information['upper_bound']

    return q_j, obj_func


