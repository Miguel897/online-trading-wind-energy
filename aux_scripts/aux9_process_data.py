
import os
import pandas as pd
from dataclasses import dataclass


def process_entsoe_data():
    @dataclass
    class Label:
        # mtu: str = 'MTU (UTC)'
        # w_on1: str = 'Generation - Wind Onshore  [MW] Day Ahead/ BZN|DK1'
        # w_off1: str = 'Generation - Wind Offshore  [MW] Day Ahead/ BZN|DK1'
        # w_on2: str = 'Generation - Wind Onshore  [MW] Day Ahead/ BZN|DK2'
        # w_off2: str = 'Generation - Wind Offshore  [MW] Day Ahead/ BZN|DK2'
        mtu: str = 'MTU'
        w_on1: str = 'Wind Onshore  - Actual Aggregated [MW]'
        w_off1: str = 'Wind Offshore  - Actual Aggregated [MW]'
        w_on2: str = 'Wind Onshore  - Actual Aggregated [MW]'
        w_off2: str = 'Wind Offshore  - Actual Aggregated [MW]'
        dk1: str = 'DK1'
        dk2: str = 'DK2'


    csv_folder = r'C:\DATOS\01_CON_RESPALDO\pycharm-workspace\doctorado\20210920_online_learning\csv'

    files = [f for f in os.listdir(csv_folder) if (f[-4:] == '.csv') and ((f[:3] == Label.dk1))]
    columns = [Label.mtu, Label.w_on1, Label.w_off1]
    # mapper = {Label.mtu: 'UTC', Label.w_on1: 'DK1_won_da', Label.w_off1: 'DK1_woff_da'}
    mapper = {Label.mtu: 'UTC', Label.w_on1: 'DK1_won_real', Label.w_off1: 'DK1_woff_real'}
    container = []

    for file in files:
        df = pd.read_csv(os.path.join(csv_folder, file))
        df = df[columns]
        df.rename(columns=mapper, inplace=True)
        container.append(df)

    df1 = pd.concat(container, axis=0)

    files = [f for f in os.listdir(csv_folder) if (f[-4:] == '.csv') and ((f[:3] == Label.dk2))]
    columns = [Label.w_on2, Label.w_off2]
    # mapper = {Label.mtu: 'UTC', Label.w_on2: 'DK2_won_da', Label.w_off2: 'DK2_woff_da'}
    mapper = {Label.mtu: 'UTC', Label.w_on1: 'DK2_won_real', Label.w_off1: 'DK2_woff_real'}

    container = []

    for file in files:
        df = pd.read_csv(os.path.join(csv_folder, file))
        df = df[columns]
        df.rename(columns=mapper, inplace=True)
        container.append(df)

    df2 = pd.concat(container, axis=0)

    df = pd.concat([df1, df2], axis=1)
    df['UTC'] = [v[0] for v in df['UTC'].str.split(' - ')]
    df['UTC'] = df['UTC'].str.replace('.', '/')
    # df = df[0:6552]  # 6551 2021-09-30 23-00
    # df.to_csv('csv/forecast_entsoe.csv', index=False)
    df.to_csv('csv/real_entsoe.csv', index=False)
    print('done')

# process_entsoe_data()
#
# re = pd.read_csv('csv/real_entsoe.csv')
# fo = pd.read_csv('csv/forecast_entsoe.csv')
# fo.drop(columns='UTC', inplace=True)
# entsoe = pd.concat([re, fo], axis=1)
# entsoe.to_csv('csv/entsoe.csv', index=False)


def process_energinet_data():
    rt = pd.read_csv('csv/realtime.csv')
    rt['HourUTC'] = pd.to_datetime(rt['HourUTC'], format='%Y-%m-%dT%H:%M:%S+00:00')
    # rt['HourUTC'] = rt['HourUTC'].dt.strftime('%d/%m/%Y %H:%M')
    rt = rt[rt['PriceArea'] == 'DK1']
    rt.reset_index(drop=True, inplace=True)
    rt.drop(columns='PriceArea', inplace=True)
    rt.set_index(keys=['HourUTC'], inplace=True)
    rt = rt.round(decimals=2)
    # rt.set_index(keys=['HourUTC', 'PriceArea'], inplace=True)
    # rt2 = rt.unstack(level=-1)

    sp = pd.read_csv('csv/spot.csv')
    # sp.rename(mapper={'HourUTC': 'HourUTCsp'}, inplace=True)
    # sp.drop(columns='HourUTC', inplace=True)
    sp['HourUTC'] = pd.to_datetime(sp['HourUTC'], format='%Y-%m-%dT%H:%M:%S+00:00')
    # sp['HourUTC'] = sp['HourUTC'].dt.strftime('%d/%m/%Y %H:%M')
    sp.set_index(keys=['HourUTC'], inplace=True)
    sp = sp.round(decimals=2)

    prices = pd.concat([rt, sp], axis=1)
    # prices = prices.iloc[::-1]
    prices.reset_index(drop=False, inplace=True)
    prices['HourUTC'] = prices['HourUTC'].dt.strftime('%d/%m/%Y %H:%M')

    row_nan = prices.isnull().any(axis=1)
    rows_with_NaN = prices[row_nan].copy()
    rows_with_NaN['BalancingPowerPriceUpEUR'] = rows_with_NaN['SpotPriceEUR']
    rows_with_NaN['BalancingPowerPriceDownEUR'] = rows_with_NaN['SpotPriceEUR']
    prices = prices.fillna(rows_with_NaN)

    prices.rename(columns={'HourUTC': 'HourUTCpr'}, inplace=True)
    entsoe = pd.read_csv('csv/entsoe.csv')
    final = pd.concat([prices, entsoe], axis=1)
    final.to_csv('csv/final.csv', index=False)
    print('done')

# process_energinet_data()

df1 = pd.read_csv(r'C:\DATOS\01_CON_RESPALDO\Input\20210929_online\dataset_DK1_wind_entsoe_2015_2021.csv')
df2 = pd.read_csv(r'C:\DATOS\01_CON_RESPALDO\Input\20210929_online\final.csv', index_col=0)
df2.drop(columns='HourUTCpr', inplace=True)
df2.rename(columns={'UTC': 'TimeUTC'}, inplace=True)
df1 = df1[:34920].round(decimals=2)
df3 = pd.concat([df1, df2], axis=0)
# df3 = df3[:24670]
# row_nan = df3.isnull().any(axis=1)
# df3[row_nan]
df3.to_csv('csv/dataset_DK1_wind_entsoe_2015_2021_2.csv', index=False)
print('done')
