import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import process_time
from os.path import join
from auxiliary.functions_standard import (
    parallel_half_space_projection,
)
from functions_core import (
    load_data, compute_fixed_action_q, sigmoidp, exp_func_01, exp_func_03
)
from config2b_online_toy import Label, Setting


class NVOnline:

    steepest_method = 'steepest'
    newton_method = 'newton'
    ada_grad = 'ada_grad'
    valid_methods = ['steepest', 'newton', 'ada_grad']

    def __init__(self,  x, E, psi_p, psi_m, alpha, memory_length, prices_mode,
                 q_0=None, lamb=1, eta=1., verbose_step=10):

        self.x = x

        if q_0 is None:
            q_0 = self.initialize_q()

        self.q = q_0.copy(deep=True)
        self.E = E
        self.psi_p = psi_p
        self.psi_m = psi_m
        self.cost_series = []
        self.alpha = alpha
        self.eta = eta
        self.squared_gradient = np.NAN
        self.lamb = lamb
        self.gradient = np.NAN
        self.hessian = np.NAN
        self.solve_method = None
        self.sample_size = len(self.E) - Setting.data_offset
        self.memory_length = memory_length
        self.memory_points = []
        self.chunk_gen = None
        self.prices_mode = prices_mode
        self.verbose_step = verbose_step
        self.computation_statistics = {}
        self.q_historical = [q_0]
        self.gradient_record = []
        self.bench_cost = None
        self.project_q = Setting.project_q
        self.regret = None

    @staticmethod
    def dot(x, q):
        return float(np.squeeze(x.reshape((1, -1)) @ q.values.reshape((-1, 1))))

    def initialize_q(self):
        # q_0 = [1e-2 for _ in range(len(Setting.feature_case))]
        # q_0 = [1e-2 for _ in self.x.columns]
        q_0 = pd.DataFrame({c: np.array([1e-2]) for c in self.x.columns})
        # q_0 = pd.DataFrame([1e-2, 1, 1e-2, 1.11, -1.65, 0.12], columns=self.x.columns)
        q_0[Label.dk1da] = 1
        return q_0

    def evaluate_true_cost(self, q):
        # q is a list

        start = Setting.data_offset
        E_d = (q * self.x[start:].reset_index(inplace=False, drop=True)).sum(axis=1)
        E_d[E_d < 0] = 0
        E_d[E_d > Setting.wind_capacity] = Setting.wind_capacity
        psi_p = self.psi_p[start:]
        psi_m = self.psi_m[start:]
        E = self.E[start:]

        cost = np.array([pp * max(0, e - e_d) + pm * max(0, e_d - e)
                              for e, e_d, pp, pm in zip(E, E_d, psi_p, psi_m)])
        return cost

    def evaluate_q_cost(self, q):
        self.cost_series.append(self.evaluate_true_objective_function(q))

    def evaluate_true_objective_function(self, q):
        x, e, pp, pm = self.last_memory_point()
        e_hat = max(0, min(Setting.wind_capacity, self.dot(x, q)))
        true_cost = pp * max(0, e - e_hat) + pm * max(0, e_hat - e)
        return true_cost

    def evaluate_smooth_objective_function(self, q):
        x, e, pp, pm = self.last_memory_point()
        smooth_cost = pp * (e - self.dot(x, q)) + self.alpha * (pp + pm) * exp_func_01((e - self.dot(x, q)) / self.alpha)
        return smooth_cost

    def compute_benchmark_cost(self):
        start = Setting.data_offset
        aol_series = np.array(
            [pp * max(0, e - x) + pm * max(0, x - e)
             for x, e, pp, pm in zip(self.x[Label.dk1da].values[start:], self.E[start:],
                                     self.psi_p[start:], self.psi_m[start:])])
        self.bench_cost = aol_series

        return aol_series

    def compute_gradient(self, q, method='newton', update_gradient=True):

        if method == self.steepest_method or method == self.ada_grad:
            if not Setting.subgradient:
                if self.prices_mode == Label.bidding_mode:
                    gradient = 1 / self.memory_length * np.array([x * (-pp + (pp + pm) * sigmoidp((e - self.dot(x, q)) / self.alpha))
                                    for x, e, pp, pm in self.memory_points]).sum(axis=0).reshape((1, -1))
                elif self.prices_mode == Label.forecasting_mode:
                    gradient = 1 / self.memory_length * np.array([x * (-1 + (1 + 1) * sigmoidp((e - self.dot(x, q)) / self.alpha))
                                    for x, e, _, _ in self.memory_points]).sum(axis=0).reshape((1, -1))
                elif self.prices_mode == Label.smoothing_mode:
                    # gradient = 1 / self.memory_length * np.array([x * (-(1 + pp) + (1 + 1 + pp + pm) * sigmoidp((e - self.dot(x, q)) / self.alpha))
                    #                 for x, e, pp, pm in self.memory_point]).sum(axis=0).reshape((1, -1))
                    gradient = 1 / self.memory_length * np.array([x * (-((1 - Setting.mu) + Setting.mu * pp)
                                                                       + (2 * (1 - Setting.mu) + Setting.mu * (pp + pm)) * sigmoidp((e - self.dot(x, q)) / self.alpha))
                                                                       for x, e, pp, pm in self.memory_points]).sum(axis=0).reshape((1, -1))
            else:
                if self.prices_mode == Label.bidding_mode:
                    gradient = 1 / self.memory_length * np.array([-pp * x if (e - self.dot(x, q)) >= 0 else pm * x
                                    for x, e, pp, pm in self.memory_points]).sum(axis=0).reshape((1, -1))
                elif self.prices_mode == Label.forecasting_mode:
                    gradient = 1 / self.memory_length * np.array([-x if (e - self.dot(x, q)) >= 0 else x
                                    for x, e, pp, pm in self.memory_points]).sum(axis=0).reshape((1, -1))
                elif self.prices_mode == Label.smoothing_mode:
                    gradient = 1 / self.memory_length * np.array([-x * ((1 - Setting.mu) + Setting.mu * pp) if (e - self.dot(x, q)) >= 0 else x * ((1 - Setting.mu) + Setting.mu * pm)
                                    for x, e, pp, pm in self.memory_points]).sum(axis=0).reshape((1, -1))

            # if len(x.shape) == 1:
            #     gradient = pd.DataFrame((aux * x).reshape((1, -1)), columns=self.x.columns)
            # else:
            gradient = pd.DataFrame(gradient, columns=self.x.columns)
        elif method == self.newton_method:
            self.compute_gradient(q, method=self.steepest_method, update_gradient=True)

            try:
                self.compute_hessian_matrix(q, update_hessian=True)
                gradient = np.linalg.inv(self.hessian) @ self.gradient.values.transpose()
                gradient = pd.DataFrame(gradient.transpose(), columns=self.gradient.columns)
            except np.linalg.LinAlgError:
                print('Broken. Check if eta is computed first.')
                gradient, update_gradient = self.eta * self.gradient, False
                print('Warning Hessian not invertible. Using steepest step instead.')

        if update_gradient:
            self.gradient = gradient

        return gradient

    def compute_hessian_matrix(self, q, update_hessian=True):

        if self.prices_mode == Label.bidding_mode:
            c_vec = 1 / self.memory_length * np.array([(pp + pm) / self.alpha * exp_func_03(e - self.dot(x, q), self.alpha, Setting.wind_capacity)
                                                       for x, e, pp, pm in self.memory_points])
        elif self.prices_mode == Label.forecasting_mode:
            c_vec = 1 / self.memory_length * np.array([(1 + 1) / self.alpha * exp_func_03(e - self.dot(x, q), self.alpha, Setting.wind_capacity)
                                                       for x, e, _, _ in self.memory_points])
        x_mem = [x for x, _, _, _ in self.memory_points]
        hessian_t = [c_t * x_t.reshape((-1, 1)) @ x_t.reshape((1, -1)) for x_t, c_t in zip(x_mem, c_vec)]
        hessian = np.array(hessian_t).sum(axis=0)
        # rank = np.linalg.matrix_rank(hessian)
        # x, e, pp, pm = self.memory_point[0]
        # print(self.memory_point[0])
        # print('The rank of the Hessian is: ', rank)

        if update_hessian:
            self.hessian = hessian

        return hessian

    def update_eta(self, i):

        if Setting.dynamic_eta:
            self.eta = max(Setting.eta_0 * (i + 1) ** -.5, Setting.eta_min)
            if (Setting.eta_0 * (i + 1) ** -.5 < Setting.eta_min) and not Setting.eta_min_flag:
                Setting.eta_min_flag = True
                print(f'Next eta value: {Setting.eta_0 * (i + 1) ** -.5}, Eta min: {Setting.eta_min}, Iteration: {i}')
        if Setting.q_based_eta:
            def determine_eta(t):
                ttl = list(range(24 * 30 * 0, 24 * 30 * 60, 24 * 30 * 3))
                tth = list(range(24 * 30 * 3, 24 * 30 * 63, 24 * 30 * 3))
                delta_q = [1, 1.462261586, 1.966316146, 3.311858429, 5.36771335, 5.71667848, 7.553885452, 7.702543107,
                           9.885127495, 10.29614072, 14.60193805, 17.18122458, 18.15978875, 18.27607325, 19.05973788,
                           23.94371785, 27.23957044, 28.58077953, 28.97140231, 29.13824226]  # , 30.11823863]
                for v in zip(ttl, tth, delta_q):
                    if v[0] <= t < v[1]:
                        return Setting.eta_0 * (v[2] / (t + 1)) ** .5
                    else:
                        return Setting.eta_0 * (30.11823863 / (t + 1)) ** .5

            self.eta = determine_eta(i)
        if Setting.solver_method == self.ada_grad:
            if self.squared_gradient is np.nan:
                self.squared_gradient = (1 - Setting.ada_delta_ro) * self.gradient ** 2
            else:
                self.squared_gradient = Setting.ada_delta_ro * self.squared_gradient + (1 - Setting.ada_delta_ro) * self.gradient ** 2

            self.eta = Setting.eta_0 * (self.squared_gradient + Setting.ada_delta_eps) ** -.5

    def update_q(self):
        if self.solve_method == self.steepest_method:
            self.q -= self.eta * self.gradient
        if self.solve_method == self.newton_method:
            self.q -= self.gradient
        if self.solve_method == self.ada_grad:
            self.q -= self.eta * self.gradient

        if self.project_q:
            x, _, _, _ = self.last_memory_point()
            q = parallel_half_space_projection(np.squeeze(self.q.values), x, lh=0, rh=Setting.wind_capacity)
            self.q = pd.DataFrame([q], columns=self.q.columns)

    def chunk_generator(self):
        m = self.memory_length
        # start = 100
        start = Setting.data_offset
        end = self.sample_size + Setting.data_offset
        for i in range(start + 1, end + 1):
            yield [(x, e, pp, pm) for x, e, pp, pm in
                              zip(self.x.values[i-m:i, :], self.E[i-m:i], self.psi_p[i-m:i], self.psi_m[i-m:i])]
        # Si start = 100 empieza en el pto 101 y coge hacia atrás los ptos necesarios i.e. m=3, (101, 100, 99).
        # i llega hasta self.sample_size -1 para acceder al último pto se necesita [self.sample_size-1: self.sample_size]
        # por eso es necesario sumar 1

    def store_step_values(self):
        # self.q_historical.append(self.q)
        # self.q_historical.append(self.q.copy(deep=True))
        self.q_historical.append(pd.DataFrame(self.q.values, columns=self.q.columns))
        self.gradient_record.append(self.gradient)

    def initialize_memory(self):
        self.chunk_gen = self.chunk_generator()

    def update_memory(self):
        self.memory_points = next(self.chunk_gen)

    def last_memory_point(self):
        return self.memory_points[-1]

    def online_bidding(self, method):

        if method not in self.valid_methods:
            raise ValueError('Invalid method.')

        self.solve_method = method
        self.computation_statistics['start'] = process_time()
        self.initialize_memory()
        for i in range(self.sample_size):
            if i % self.verbose_step == 0 and i > 0:
                print('The value of q in iteration {} is {}'.format(i + 1, list(self.q.values)))
                print('Gradient is: {}'.format(list(self.gradient.values)))
            self.update_memory()
            self.evaluate_q_cost(self.q)
            self.compute_gradient(self.q, method=self.solve_method, update_gradient=True)
            self.update_eta(i=i)
            self.update_q()
            self.store_step_values()

        bench_cost = self.compute_benchmark_cost()
        self.cost_series = np.array(self.cost_series)

        self.computation_statistics['end'] = process_time()
        self.computation_statistics['elapsed_time'] = \
            self.computation_statistics['end'] - self.computation_statistics['start']
        self.computation_statistics['n_samples'] = self.sample_size
        self.computation_statistics['eta'] = self.eta
        self.computation_statistics['alpha'] = self.alpha
        self.computation_statistics['update_method'] = self.solve_method
        self.computation_statistics['gradient'] = self.gradient
        self.computation_statistics['OL_cost'] = np.sum(self.cost_series) / self.sample_size
        self.computation_statistics['FO_cost'] = np.sum(bench_cost) / self.sample_size
        self.computation_statistics['final_q'] = self.q

    def print_computation_report(self):
        if len(self.computation_statistics) == 0:
            raise ValueError('No statistics recorded.')

        summary = [
            'Solver summary statistics:',
            '--------------------------',
            # 'Starting time: {}'.format(self.computation_statistics['start']),
            # 'Ending time: {}'.format(self.computation_statistics['end']),
            'Elapsed time: {}'.format(self.computation_statistics['elapsed_time']),
            'Solver method: {}'.format(self.computation_statistics['update_method']),
            'Sample size: {}'.format(self.computation_statistics['n_samples']),
            'alpha parameter: {}'.format(self.computation_statistics['alpha']),
            # 'Gradient value:\n {}'.format(self.computation_statistics['gradient'].to_string(index=False)),
            # 'Final q:\n {}'.format(self.computation_statistics['final_q'].to_string(index=False)),
            'Benchmark cost: {}'.format(self.computation_statistics['FO_cost']),
            'Avg. cost: {}'.format(self.computation_statistics['OL_cost']),
         ]

        if self.solve_method == NVOnline.steepest_method:
            summary.append(
                'Eta parameter: {}'.format(self.computation_statistics['eta']),
            )

        print('\n' + '\n'.join(summary) + '\n')

    def export_results(self):
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        q_df = pd.concat(self.q_historical).reset_index(drop=True)
        gradient_df = pd.concat(self.gradient_record).reset_index(drop=True)
        start = Setting.data_offset
        E_d = (q_df.iloc[:-1, :] * self.x[start:].reset_index(inplace=False, drop=True)).sum(axis=1)
        E_d[E_d < 0] = 0
        E_d[E_d > Setting.wind_capacity] = Setting.wind_capacity
        E_d.index = self.x[start:].index
        E_d.name = 'E_D'
        psi_p = self.psi_p[start:]
        psi_p.name = 'psi_p'
        psi_m = self.psi_m[start:]
        psi_m.name = 'psi_m'
        E = self.E[start:]
        x = self.x.iloc[start:, :]

        cost = pd.Series(self.cost_series, index=E_d.index, name='Cost')
        bnch_cost = pd.Series(self.bench_cost, index=E_d.index, name='BCost')

        q_fixed, cost_fixed = compute_fixed_action_q(E, psi_p, psi_m, x, (0, Setting.wind_capacity))
        # q_fixed = [8.062370463049456, 0.8525360309677276, 0.06398442124976188, 7.238483110156916, -0.442520053664792, 254.109049776738]
        # cost_fixed = 249.762799129683
        E_fixed = (x * q_fixed).sum(axis=1)
        E_fixed.name = 'E_fixed'
        E_fix_cost = pd.Series(self.evaluate_true_cost(q_fixed), index=E_d.index, name='E_fix_cost')
        rounds = range(1, len(E_fix_cost) + 1)
        regret = (cost.cumsum() - E_fix_cost.cumsum()) / rounds
        regret.name = 'regret'
        self.regret = regret

        report = pd.DataFrame({'Setting': [
            'Elapsed time',
            'Solver method',
            'Sample size:',
            'Alpha:',
            'Eta_0:',
            'Eta:',
            'mu:',
            'Memory len:',
            'Price Mode:',
            'FO MAE:',
            'OL MAE:',
            'FO RMSE:',
            'OL RMSE:',
            'FO cost:',
            'OL cost:',
            'Regressors:',
            'q_fixed:',
            'fixed cost:',
            'Avg. regret:',
        ], 'value': [
            self.computation_statistics['elapsed_time'],
            self.solve_method,
            self.sample_size,
            self.alpha,
            Setting.eta_0,
            self.eta,
            Setting.mu,
            self.memory_length,
            self.prices_mode.lower(),
            mean_absolute_error(E, self.x[Label.dk1da][start:]),
            mean_absolute_error(E, E_d),
            mean_squared_error(E, self.x[Label.dk1da][start:]),
            mean_squared_error(E, E_d),
            self.computation_statistics['FO_cost'],
            self.computation_statistics['OL_cost'],
            self.x.shape[1],
            q_fixed,
            cost_fixed,
            self.computation_statistics['OL_cost'] - cost_fixed,
        ]})

        writer = pd.ExcelWriter(join(Setting.sim_path, Setting.timestamp + 'results' + '.xlsx'), engine='xlsxwriter')
        report.to_excel(writer, sheet_name='summary_report', index=False)
        (pd.concat([psi_p, psi_m, x, E, x[Label.dk1da], E_fixed, E_d, bnch_cost, E_fix_cost, cost, regret], axis=1)).to_excel(writer, sheet_name='source_data')
        q_df.to_excel(writer, sheet_name='Regressors')
        gradient_df.to_excel(writer, sheet_name='Gradient')
        writer.save()

    def plot_iteration_evolution(self):
        # print(self.q_historical)
        # q_vector = [pd.DataFrame(q) for q in self.q_historical]
        # steps = range(1, len(self.q_historical) + 1)
        plt.plot([0, len(self.regret) + 2], [0, 0], color='black', linestyle='dashed')
        plt.plot(self.regret.reset_index(inplace=False, drop=True))
        plt.xlabel('Iterations')
        plt.ylabel('Avg. Regret')
        # plt.show()
        fig = plt.gcf()
        fig.savefig(join(Setting.sim_path, Setting.timestamp + 'regret' + '.png'))

        # ax = plt.subplot(3,1)
        # ax[0].plot(steps, true_objective, color='blue', label='True obj.')
        # plt.plot(iterations, smooth_objective, color='cyan', label='Smooth obj.')
        # plt.xticks(np.arange(min(iterations) - 1, max(iterations) + 1, 1.0))
        # plt.xlabel('Iterations')
        # plt.ylabel('Obj. value')
        # plt.title(r'Convergence for $\alpha = $' + str(self.alpha))
        # plt.legend()
        # plt.show()
        # print(true_objective)
        # print(smooth_objective)
        # print([smooth_objective[-1] < x for x in smooth_objective])
        # print([true_objective[-1] < x for x in true_objective])


def main_03_toy_step():
    wind, b_data, h_data, x_data = load_data(Label, Setting, add_ones=True, case='synthetic')
    nv_online = NVOnline(x=x_data, E=wind, psi_p=b_data, psi_m=h_data, alpha=Setting.alpha, prices_mode=Setting.mode,
                         memory_length=Setting.memory_length, lamb=1, eta=Setting.eta, verbose_step=Setting.verbose_steps)

    nv_online.online_bidding(method=Setting.solver_method)
    nv_online.print_computation_report()
    nv_online.export_results()
    nv_online.plot_iteration_evolution()


if __name__ == '__main__':
    main_03_toy_step()
