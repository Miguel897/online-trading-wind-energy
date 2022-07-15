import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from time import process_time
from dataclasses import dataclass
from functions_standard import create_directory, time_it, parallel_half_space_projection
from functions_core import (
    load_data_forecast, compute_fixed_action_q
)
from config9_forecasting import Label, Setting, platform_context, script_simulation_dict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

y_re, y_da, x_data = load_data_forecast(Label, Setting, add_ones=False)

tr_start = datetime.datetime.strptime('2015-01-08', '%Y-%m-%d')
# tr_start = datetime.datetime.strptime('2019-01-08', '%Y-%m-%d')
tr_end = datetime.datetime.strptime('2015-07-01', '%Y-%m-%d')
# tr_end = datetime.datetime.strptime('2015-12-01', '%Y-%m-%d')
# tr_end = datetime.datetime.strptime('2019-07-01', '%Y-%m-%d')
# tr_end = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')
mask_tr = (y_re.index >= tr_start) & (y_re.index < tr_end)
mask_ts = y_re.index >= tr_end
mask_l1 = y_re.index >= (tr_end - datetime.timedelta(hours=1))
mask_l1[-1] = False

y_re_tr = y_re[mask_tr].values
y_re_ts = y_re[mask_ts].values

y_l1_ts = y_re[mask_l1].values

y_da_tr = y_da[mask_tr].values
y_da_ts = y_da[mask_ts].values

x_tr = x_data[mask_tr].values
x_ts = x_data[mask_ts].values

reg = LinearRegression().fit(x_tr, y_re_tr.reshape((-1, 1)))

y_pr_ts = np.squeeze(reg.predict(x_ts))

print('RMSE original: ', round((mean_squared_error(y_re_ts, y_da_ts))**0.5, 2))
print('RMSE l1: ', round((mean_squared_error(y_re_ts, y_l1_ts))**0.5, 2))
print('RMSE predicted: ', round((mean_squared_error(y_re_ts, y_pr_ts))**0.5, 2))
# x_tr = x_data
# train_test_split(shuffle=False)
# wind
print('Done')

# df_ts = pd.DataFrame({'y_da': y_da_ts, 'y_pr': y_pr_ts, 'y_l1': y_l1_ts})
df = pd.DataFrame({Label.dk1da: y_pr_ts}, index=x_data[mask_ts].index)
df.to_csv(f'csv/{Label.dk1da}.csv')

# plt.plot(y_da_ts[0:2400], label='y_da')
# plt.plot(y_pr_ts[0:2400], label='y_pr')
# plt.plot(y_re_ts[0:2400], label='y_dr')
# plt.legend()
# plt.show()