import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from auxiliary.functions_standard import time_it
from dataclasses import dataclass
from optimization_lp import create_bigadata_nv_model, solve_optimization_model_direct, standard_solving_configuration
from functions_core import (
    load_data,
)
# from config_lp2 import Label, Setting
from config2_online_bidding import Label, Setting


@time_it
def routine01(vebose=True):
    standard_solving_configuration['solver_factory_options']['options']['simplex_tolerance_optimality'] = 1e-4
    # standard_solving_configuration['solver_factory_options']['options']['lpmethod'] = 2

    # wind, b_data, h_data, x_data = load_data(Label, Setting)
    # sample_size = x_data.shape[0]

    # Usar esto para obtener el valor insample optimo de la grafica de steps
    # from config3b_lp_toy import Setting
    # Quitar el offset del config --> data_offset: int = 24 * 0
    # wind, b_data, h_data, x_data = load_data(Label, Setting, add_ones=False, case='synthetic')

    end_date = []
    # Setting.complete_data_set = (datetime.datetime(year=2016, month=1, day=1, hour=0),
    Setting.complete_data_set = (datetime.datetime(year=2016, month=7, day=1, hour=0),
                                 datetime.datetime(year=2021, month=6, day=5, hour=0))
                                 # datetime.datetime(year=2023, month=1, day=1, hour=0))
    wind, b_data, h_data, x_data = load_data(Label, Setting, add_ones=True, case='real')
    s, e = 0, None
    # s, e = 24, 24 * 30 + 24
    # s = 24 * 30 * 2
    # e = 24 * 30 * 2
    # e = 24 * 30 * 4
    wind = wind[s:e]
    b_data = b_data[s:e]
    h_data = h_data[s:e]
    x_data = x_data[s:e]

    data = {
        'y_i': wind,
        'x_ij': x_data,
        'b_i': b_data,
        'h_i': h_data,
        'lamb': 0,
        'y_bounds': (0, Setting.wind_capacity),
        'q_j_bounds': (None, None),
    }

    option = 'obtain_single_q_h'
    # option = 'determine_set_diameter'
    if option == 'obtain_single_q_h':
        model = create_bigadata_nv_model(data)
        solved_model, solver_status, solver_additional_information = solve_optimization_model_direct(model, standard_solving_configuration)
        q_j = list(solved_model.q_j[j].value for j in solved_model.j)
        obj_func = solver_additional_information['upper_bound']
        print('Value of q_j:')
        print(q_j)
        # solved_model.q_j.pprint()
        print('Avg. value of the obj. func:')
        print(solver_additional_information['upper_bound'])
        print('Done!')

        return q_j, obj_func

    elif option == 'determine_set_diameter':
        from optimization_lp import create_feasible_set_diameter_model
        model = create_feasible_set_diameter_model(data)
        standard_solving_configuration['solver_factory_options']['options']['optimalitytarget'] = 3
        solved_model, solver_status, solver_additional_information = solve_optimization_model_direct(
            model, standard_solving_configuration)
        q1 = list(solved_model.q1[j].value for j in solved_model.j)
        q2 = list(solved_model.q2[j].value for j in solved_model.j)
        print(q1)
        print(q2)
        from scipy.spatial.distance import euclidean
        distance = euclidean(q1, q2)
        print(distance)
        print('Done!')
        return distance, q1, q2

    # offers = (x_data * q_j).sum(axis=1)
    # offers.to_csv('data_offers.csv')


def routine01b(wind, b_data, h_data, x_data, y_bounds):
    standard_solving_configuration['solver_factory_options']['options']['simplex_tolerance_optimality'] = 1e-2
    standard_solving_configuration['solver_factory_options']['options']['lpmethod'] = 2  # Dual simplex

    data = {
        'y_i': wind,
        'x_ij': x_data,
        'b_i': b_data,
        'h_i': h_data,
        'lamb': 0,
        'y_bounds': y_bounds,
        'q_j_bounds': (None, None),
    }

    model = create_bigadata_nv_model(data)
    solved_model, solver_status, solver_additional_information = solve_optimization_model_direct(
        model, standard_solving_configuration)
    q_j = list(solved_model.q_j[j].value for j in solved_model.j)
    obj_func = solver_additional_information['upper_bound']

    return q_j, obj_func


def routine01c(vebose=True):
    standard_solving_configuration['solver_factory_options']['options']['simplex_tolerance_optimality'] = 1e-4
    # standard_solving_configuration['solver_factory_options']['options']['lpmethod'] = 2

    def get_serie_interval(interval, *args):
        return (serie[(serie.index >= interval[0]) & (serie.index < interval[1])] for serie in args)

    @dataclass
    class r_label:
        days: int = 30
        static: str = 'incremental_static_q'
        dynamic: str = 'sequential_dynamic_q'
        # offer: str = '20220303_olnv_offer.csv'
        # offer: str = '20220310_olnv_offer.csv'
        offer: str = '20220329_ada_grad_mu07_eta1e-3.csv'
        result_file: str = 'result_data_{}m.csv'

    class r_opt:
        q_h: str = r_label.dynamic  # <-------
        # q_h: str = r_label.static
        compute_regret: bool = True
        n_months: int = 60
        # month_step: int = 3
        # n_steps: int = int(60 / 3)
        # month_step: int = 6
        # n_steps: int = int(60 / 6)
        month_step: int = 12
        n_steps: int = int(60 / 12)

    start_date = datetime.datetime(year=2016, month=7, day=1, hour=0)
    # end_date = start_date + datetime.timedelta(days=r_label.days * r_opt.n_months)
    summary_result_container = {
        'months': [], 'Bcost': [], 'E_fix_cost': [], 'E_fix_cost_p': [], 'Cost': [], 'Cost_p': [], 'Regret_t': [], 'q_h': []}

    if r_opt.compute_regret:
        olnv_offer_raw = pd.read_csv(os.path.join(Setting.input_folder_path, r_label.offer), index_col=0, sep=';', decimal=',')
        olnv_offer_raw['TimeUTC'] = pd.to_datetime(olnv_offer_raw['TimeUTC'], format='%Y-%m-%d %H:%M:%S')
        olnv_offer_raw.set_index(['TimeUTC'], inplace=True, drop=True)
        olnv_offer_raw = olnv_offer_raw['E_D']
        olnv_offer_raw.name = 'E_D'

    if r_opt.q_h == r_label.static:
        start_dates = [start_date for _ in range(0, r_opt.n_months, r_opt.month_step)]
    elif r_opt.q_h == r_label.dynamic:
        hourly_result_container = []
        start_dates = [start_date + datetime.timedelta(days=r_label.days * m)
                       for m in range(0, r_opt.n_months, r_opt.month_step)]
    else:
        raise ValueError('Not implemented')

    n_months = list(range(r_opt.month_step, r_opt.n_months + r_opt.month_step, r_opt.month_step))
    end_dates = [start_date + datetime.timedelta(days=r_label.days * m) for m in n_months]
    end_date = end_dates[-1]
    intervals = zip(start_dates, end_dates, n_months)

    deltah = datetime.timedelta(hours=1)  # I need hour -1 to obtain the lags of the penalties
    # Setting.complete_data_set = (datetime.datetime(year=2016, month=1, day=1, hour=1) - deltah,
    Setting.complete_data_set = (start_date - deltah, end_date)
    wind_raw, b_data_raw, h_data_raw, x_data_raw = load_data(Label, Setting, add_ones=True, case='real')
    b_data_raw.name = 'psi_p'
    h_data_raw.name = 'psi_m'

    for interval in intervals:
        print(f'Executing interval {interval[2]}m ...')
        if r_opt.compute_regret:
            olnv_offer, = get_serie_interval(interval, olnv_offer_raw)
            # olnv_offer_raw[(olnv_offer_raw.index > interval[0]) & (olnv_offer_raw.index < interval[1])]

        wind, b_data, h_data, x_data = get_serie_interval(interval, wind_raw, b_data_raw, h_data_raw, x_data_raw)

        data = {
            'y_i': wind,
            'x_ij': x_data,
            'b_i': b_data,
            'h_i': h_data,
            'lamb': 0,
            'y_bounds': (0, Setting.wind_capacity),
            'q_j_bounds': (None, None),
        }

        model = create_bigadata_nv_model(data)
        solved_model, solver_status, solver_additional_information = solve_optimization_model_direct(
            model, standard_solving_configuration)
        q_j = list(solved_model.q_j[j].value for j in solved_model.j)
        obj_func = solver_additional_information['upper_bound']

        fo_offer = x_data['DK1_won_da']
        fix_offer = (x_data * np.array(q_j)).sum(axis=1).apply(lambda x: max(0, min(Setting.wind_capacity, x)))
        fix_cost = b_data * np.maximum(0, wind - fix_offer) + h_data * np.maximum(0, fix_offer - wind)
        fo_cost = b_data * np.maximum(0, wind - fo_offer) + h_data * np.maximum(0, fo_offer - wind)
        if r_opt.compute_regret:
            olnv_cost = b_data * np.maximum(0, wind - olnv_offer) + h_data * np.maximum(0, olnv_offer - wind)
            olnv_cost.name = 'Cost'

        fo_offer.name = 'DK1_won_da'
        fo_cost.name = 'BCost'
        fix_offer.name = 'E_fix'
        fix_cost.name = 'E_fix_cost'

        if not r_opt.compute_regret:
            columns = [wind, b_data, h_data, x_data, fo_offer, fix_offer, fo_cost, fix_cost]
        else:
            columns = [wind, b_data, h_data, x_data, fo_offer, fix_offer, olnv_offer, fo_cost, fix_cost, olnv_cost]

        hourly_results = pd.concat(columns, axis=1).round(decimals=3)

        if r_opt.q_h == r_label.dynamic:
            hourly_result_container.append(hourly_results)
            hourly_results = pd.concat(hourly_result_container)

        fo_cost_mean = hourly_results['BCost'].mean()
        fix_cost_mean = hourly_results['E_fix_cost'].mean()
        summary_result_container['months'].append(interval[2])
        summary_result_container['Bcost'].append(fo_cost_mean)
        summary_result_container['E_fix_cost'].append(fix_cost_mean)
        summary_result_container['E_fix_cost_p'].append((fo_cost_mean - fix_cost_mean) / fo_cost_mean * 100)
        if r_opt.compute_regret:
            olnv_cost_mean = hourly_results['Cost'].mean()
            summary_result_container['Cost'].append(olnv_cost_mean)
            summary_result_container['Cost_p'].append((fo_cost_mean - olnv_cost_mean) / fo_cost_mean * 100)
            summary_result_container['Regret_t'].append(olnv_cost_mean - fix_cost_mean)
        q_str = ';'.join([str(q) for q in q_j]).replace('.', ',')
        summary_result_container['q_h'].append(q_str)

        hourly_results.to_csv(os.path.join(Setting.sim_path, r_label.result_file.format(interval[2])), sep=';', decimal=',')

    # if r_opt.q_h == r_label.dynamic:
    #     df = pd.concat(hourly_result_container)
    #     df.to_csv(os.path.join(Setting.sim_path, r_label.result_file.format(intervals[-1][2])), sep=';', decimal=',')

    df = pd.DataFrame(summary_result_container)
    df.to_csv(os.path.join(Setting.sim_path, r_label.result_file.format('summary')), sep=';', decimal=',')
    # offers = (x_data * q_j).sum(axis=1)

    # if vebose:
    #     print('Value of q_j:')
    #     print(q_j)
    #     # solved_model.q_j.pprint()
    #     print('Avg. value of the obj. func:')
    #     print(solver_additional_information['upper_bound'])
    #     print('Done!')
    #
    # return q_j, obj_func


# Regression line
def routine02_plotting():
    q_j = list(m.q_j[j].value for j in m.j)
    # plt.scatter(x['E_est'].values, y, color='grey')

    x_val = [0.05 * Setting.wind_capacity, 0.95 * Setting.wind_capacity]
    y_val = [q_j[0] + q_j[1] * v for v in x_val]
    plt.plot(x_val, y_val)

    plt.legend()
    plt.show()


def routine03_plotting():
    # q_j = list(m.q_j[j].value for j in m.j)
    # plt.scatter(x['E_est'].values, y, color='grey')

    q_10 = [-6.513877097830214, 1.0158969317374131]
    q_30 = [-0.8522815811672918, 0.9670639483747466]
    q_50 = [0.12604017754549737, 0.9910724871753129]
    q_70 = [1.0327739440945933, 1.0270689127884474]
    q_90 = [5.467048702616316, 0.9885769907350571]

    for q, c, l in zip([q_10, q_30, q_50, q_70, q_90], ['lightblue', 'cyan', 'orange', 'cyan', 'lightblue'], ['0.1', '0.3', '0.5', '0.7', '0.9']):
    # for q in [q_j]:
        x_val = [0.05 * Setting.wind_capacity, 0.95 * Setting.wind_capacity]
        y_val = [q[0] + q[1] * v for v in x_val]
        plt.plot(x_val, y_val, color=c, label=l)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    pass
    routine01()
    # routine01b()
    # routine01c()
    # routine02_plotting()
    # routine03_plotting()
