from dataclasses import dataclass


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
    static: str = 'incremental_static_q'
    dynamic: str = 'sequential_dynamic_q'
    # offer: str = '20220303_olnv_offer.csv'
    # offer: str = '20220310_olnv_offer.csv'
    offer: str = '20220329_ada_grad_mu07_eta1e-3.csv'
    result_file: str = 'result_data_{}m.csv'
    days: int = 30
