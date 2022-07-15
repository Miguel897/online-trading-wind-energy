import numpy as np
import pandas as pd
import datetime
from itertools import product


cap = pd.read_csv(r'C:\DATOS\01_CON_RESPALDO\Input\20210929_online\wind_capacities.csv', sep='\t', index_col=0)
df = pd.read_csv(r'C:\DATOS\01_CON_RESPALDO\Input\20210929_online\dataset_DK1_wind_entsoe_2015_2021.csv')
df.set_index('TimeUTC', inplace=True, drop=True)
df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
cap_df = pd.DataFrame(np.ones(df.shape), columns=df.columns, index=df.index)
subfix = ('_real', '_da')

for year, area in product(cap.index, cap.columns):
    c = cap.at[year, area]
    # print(c)
    mask = (df.index >= datetime.datetime(year=year, month=1, day=1, hour=0)) & \
    (df.index <= datetime.datetime(year=year, month=12, day=31, hour=23))
    for sub in subfix:
        cap_df.loc[mask, area + sub] = c / 100

df = df / cap_df
df = df.round(decimals=2)
df.loc[df['DK1_woff_real'] > 150, 'DK1_woff_real'] = np.nan
df.loc[df['DK2_won_real'] > 150, 'DK2_won_real'] = np.nan
print(df.max())
print((df > 100).sum())
# columns = ['DK1_won_real', 'DK1_won_da', 'DK1_woff_real', 'DK1_woff_da', 'DK2_won_real', 'DK2_won_da', 'DK2_woff_real', 'DK2_woff_da']
columns = ['SpotPriceEUR', 'BalancingPowerPriceDownEUR', 'BalancingPowerPriceUpEUR']
mask = df > 100
mask.loc[:, columns] = False
df[mask] = 100
df.reset_index(inplace=True, drop=False)
df['TimeUTC'] = df['TimeUTC'].dt.strftime('%d/%m/%Y %H:%M')
df.interpolate(inplace=True, limit_direction='both')
df.to_csv(r'C:\DATOS\01_CON_RESPALDO\Input\20210929_online\dataset_DK1_wind_entsoe_2015_2021_s.csv', index=False)
print('done')
