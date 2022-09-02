import datetime
import os
import numpy as np
import pandas as pd
from functions_standard import time_it
from dataclasses import dataclass
from functions_core import (
    load_data,
    compute_optimal_lp_q,
)
# from config_lp2 import Label, Setting
from config2_online_bidding import Label, Setting


# @time_it   # To obtain the time of computing the best single q in hindsight q^H
def main_optimal_lp_q(save_offers=False, vebose=True):
    """Used to compute the best q for the toy example "Dynamic behavior" and
    the optimal hindsight q^H for the case study
    """

    # Usar esto para obtener el valor insample optimo de la grafica de steps
    # from config3b_lp_toy import Setting
    # Quitar el offset del config --> data_offset: int = 24 * 0
    # wind, psi_p, psi_m, x_data = load_data(Label, Setting, add_ones=False, case=

    # real: load real data from Energinet. Can be used for Comparing the smooth and subgradient implementations
    # synthetic: generate the synthetic of the toy example "Dynamic behavior"
    case = 'synthetic'  # real, convergence, synthetic
    if case == 'real':
        add_ones = True
    elif case == 'synthetic':
        add_ones = False

    # Setting.complete_data_set = (datetime.datetime(year=2016, month=1, day=1, hour=0),
    Setting.complete_data_set = (datetime.datetime(year=2016, month=7, day=1, hour=0),
                                 datetime.datetime(year=2021, month=6, day=5, hour=0))
                                 # datetime.datetime(year=2023, month=1, day=1, hour=0))
    wind, psi_p, psi_m, x_data = load_data(Label, Setting, add_ones=add_ones, case=case)
    s, e = 0, None
    # s, e = 24, 24 * 31
    # s, e = 24 * 30 * 2, 24 * 30 * 2
    # s, e = 24 * 30 * 2, 24 * 30 * 4
    wind = wind[s:e]
    psi_p = psi_p[s:e]
    psi_m = psi_m[s:e]
    x_data = x_data[s:e]
    y_bounds = (0, Setting.wind_capacity)

    extra_options = {
        'simplex_tolerances_optimality': '1e-4',
        'lpmethod': 2,  # Dual simplex
    }
    q_j, obj_func = compute_optimal_lp_q(wind, psi_p, psi_m, x_data, y_bounds, extra_options=extra_options)
    if save_offers:
        offers = (x_data * q_j).sum(axis=1)
        offers.to_csv('data_offers.csv')

    print('Value of q_j:\n', q_j)
    print('Avg. value of the obj. func:\n', obj_func)
    print('Done!')

    return q_j, obj_func


def compute_regret():
    """Function to compute the static and dynamic regret benchamark in the paper.
    """

    def get_serie_interval(interval, *args):
        return (serie[(serie.index >= interval[0]) & (serie.index < interval[1])] for serie in args)

    @dataclass
    class r_label:
        days: int = 30
        static: str = 'incremental_static_q'
        dynamic: str = 'sequential_dynamic_q'

    @dataclass
    class r_confg:
        offer: str = '20220329_ada_grad_mu07_eta1e-3.csv'
        result_file: str = 'result_data_{}m.csv'
        q_h: str = r_label.dynamic  # Selector r_label.static or r_label.dynamic to compute the static or dynamic regret
        compute_regret: bool = True
        n_months: int = 60
        month_step: int = 12  # In the paper 3, 6, 12 months

    start_date = datetime.datetime(year=2016, month=7, day=1, hour=0)
    summary_result_container = {
        'months': [], 'Bcost': [], 'E_fix_cost': [], 'E_fix_cost_p': [],
        'Cost': [], 'Cost_p': [], 'Regret_t': [], 'q_h': [],
    }

    if r_confg.compute_regret:
        olnv_offer_raw = pd.read_csv(os.path.join(Setting.input_folder_path, r_confg.offer), index_col=0, sep=';', decimal=',')
        olnv_offer_raw['TimeUTC'] = pd.to_datetime(olnv_offer_raw['TimeUTC'], format='%Y-%m-%d %H:%M:%S')
        olnv_offer_raw.set_index(['TimeUTC'], inplace=True, drop=True)
        olnv_offer_raw = olnv_offer_raw['E_D']
        olnv_offer_raw.name = 'E_D'

    if r_confg.q_h == r_label.static:
        start_dates = [start_date for _ in range(0, r_confg.n_months, r_confg.month_step)]
    elif r_confg.q_h == r_label.dynamic:
        hourly_result_container = []
        start_dates = [start_date + datetime.timedelta(days=r_label.days * m)
                       for m in range(0, r_confg.n_months, r_confg.month_step)]
    else:
        raise ValueError('Not implemented.')

    n_months = list(range(r_confg.month_step, r_confg.n_months + r_confg.month_step, r_confg.month_step))
    end_dates = [start_date + datetime.timedelta(days=r_label.days * m) for m in n_months]
    end_date = end_dates[-1]
    intervals = zip(start_dates, end_dates, n_months)

    deltah = datetime.timedelta(hours=1)  # I need hour -1 to obtain the lags of the penalties
    Setting.complete_data_set = (start_date - deltah, end_date)
    wind_raw, b_data_raw, h_data_raw, x_data_raw = load_data(Label, Setting, add_ones=True, case='real')
    b_data_raw.name = 'psi_p'
    h_data_raw.name = 'psi_m'

    for interval in intervals:
        print(f'Executing interval {interval[2]}m ...')
        if r_confg.compute_regret:
            olnv_offer, = get_serie_interval(interval, olnv_offer_raw)

        wind, psi_p, psi_m, x_data = get_serie_interval(interval, wind_raw, b_data_raw, h_data_raw, x_data_raw)

        y_bounds = (0, Setting.wind_capacity)
        extra_options = {
            'simplex_tolerances_optimality': '1e-4',
            'lpmethod': 2,  # Dual simplex
        }
        q_j, obj_func = compute_optimal_lp_q(wind, psi_p, psi_m, x_data, y_bounds, extra_options=extra_options)

        fo_offer = x_data['DK1_won_da']
        fix_offer = (x_data * np.array(q_j)).sum(axis=1).apply(lambda x: max(0, min(Setting.wind_capacity, x)))
        fix_cost = psi_p * np.maximum(0, wind - fix_offer) + psi_m * np.maximum(0, fix_offer - wind)
        fo_cost = psi_p * np.maximum(0, wind - fo_offer) + psi_m * np.maximum(0, fo_offer - wind)
        if r_confg.compute_regret:
            olnv_cost = psi_p * np.maximum(0, wind - olnv_offer) + psi_m * np.maximum(0, olnv_offer - wind)
            olnv_cost.name = 'Cost'

        fo_offer.name = 'DK1_won_da'
        fo_cost.name = 'BCost'
        fix_offer.name = 'E_fix'
        fix_cost.name = 'E_fix_cost'

        if not r_confg.compute_regret:
            columns = [wind, psi_p, psi_m, x_data, fo_offer, fix_offer, fo_cost, fix_cost]
        else:
            columns = [wind, psi_p, psi_m, x_data, fo_offer, fix_offer, olnv_offer, fo_cost, fix_cost, olnv_cost]

        hourly_results = pd.concat(columns, axis=1).round(decimals=3)

        if r_confg.q_h == r_label.dynamic:
            hourly_result_container.append(hourly_results)
            hourly_results = pd.concat(hourly_result_container)

        fo_cost_mean = hourly_results['BCost'].mean()
        fix_cost_mean = hourly_results['E_fix_cost'].mean()
        summary_result_container['months'].append(interval[2])
        summary_result_container['Bcost'].append(fo_cost_mean)
        summary_result_container['E_fix_cost'].append(fix_cost_mean)
        summary_result_container['E_fix_cost_p'].append((fo_cost_mean - fix_cost_mean) / fo_cost_mean * 100)

        if r_confg.compute_regret:
            olnv_cost_mean = hourly_results['Cost'].mean()
            summary_result_container['Cost'].append(olnv_cost_mean)
            summary_result_container['Cost_p'].append((fo_cost_mean - olnv_cost_mean) / fo_cost_mean * 100)
            summary_result_container['Regret_t'].append(olnv_cost_mean - fix_cost_mean)
        q_str = ';'.join([str(q) for q in q_j]).replace('.', ',')
        summary_result_container['q_h'].append(q_str)

        hourly_results.to_csv(os.path.join(Setting.sim_path, r_confg.result_file.format(interval[2])), sep=';', decimal=',')

    df = pd.DataFrame(summary_result_container)
    df.to_csv(os.path.join(Setting.sim_path, r_confg.result_file.format('summary')), sep=';', decimal=',')


if __name__ == '__main__':
    main_optimal_lp_q()
    # compute_regret()
