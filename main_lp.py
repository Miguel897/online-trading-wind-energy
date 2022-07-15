
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_standard import time_it
from dataclasses import dataclass
from optimization_lp import create_bigadata_nv_model, solve_optimization_model, standard_solving_configuration


@dataclass
class Settings:
    n_samples: int = 100
    error: float = 7.
    gen_capacity: float = 100


standard_solving_configuration['solver_factory_options']['options']['simplex_tolerance_optimality'] = 1e-2


np.random.seed(17)

y = np.random.uniform(10, 90, Settings.n_samples)
x = y + np.random.uniform(- Settings.error, Settings.error, Settings.n_samples)
x = pd.DataFrame(x, columns=['E_est'])
x['ones'] = 1
x = x[['ones', 'E_est']]
#b_i = 4 * np.ones(Settings.n_samples)
tau = 0.9
b_i = tau * np.ones(Settings.n_samples)  # psi_+
h_i = (1 - tau) * np.ones(Settings.n_samples)  # psi_-

data = {
    'y_i': y,
    'x_ij': x,
    'b_i': b_i,
    'h_i': h_i,
    'lamb': 0,
    'y_bounds': (0, Settings.gen_capacity),
    'q_j_bounds': (None, None),
}


@time_it
def routine01():
    model = create_bigadata_nv_model(data)
    solved_model, solver_status, solver_additional_information = solve_optimization_model(model, standard_solving_configuration)
    solved_model.q_j.pprint()
    print('Done!')
    return solved_model


m = routine01()


# Regression line
def routine02_plotting():
    q_j = list(m.q_j[j].value for j in m.j)
    plt.scatter(x['E_est'].values, y, color='grey')

    x_val = [0.05 * Settings.gen_capacity, 0.95 * Settings.gen_capacity]
    y_val = [q_j[0] + q_j[1] * v for v in x_val]
    plt.plot(x_val, y_val)

    plt.legend()
    plt.show()


routine02_plotting()


def routine03_plotting():
    # q_j = list(m.q_j[j].value for j in m.j)
    plt.scatter(x['E_est'].values, y, color='grey')

    q_10 = [-6.513877097830214, 1.0158969317374131]
    q_30 = [-0.8522815811672918, 0.9670639483747466]
    q_50 = [0.12604017754549737, 0.9910724871753129]
    q_70 = [1.0327739440945933, 1.0270689127884474]
    q_90 = [5.467048702616316, 0.9885769907350571]

    for q, c, l in zip([q_10, q_30, q_50, q_70, q_90], ['lightblue', 'cyan', 'orange', 'cyan', 'lightblue'], ['0.1', '0.3', '0.5', '0.7', '0.9']):
    # for q in [q_j]:
        x_val = [0.05 * Settings.gen_capacity, 0.95 * Settings.gen_capacity]
        y_val = [q[0] + q[1] * v for v in x_val]
        plt.plot(x_val, y_val, color=c, label=l)

    plt.legend()
    plt.show()


routine03_plotting()
