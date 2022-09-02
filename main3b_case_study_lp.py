import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import process_time
from os.path import join
from functions_specific import (
    load_data, compute_optimal_lp_q)
from config3b_case_study_lp import Label, Setting


class NVlp:

    def __init__(self,  x, E, psi_p, psi_m, q_0=None, verbose_step=10):

        self.x = x
        self.price_mode = Setting.price_mode

        if q_0 is None:
            q_0 = self.initialize_q()

        self.q = q_0.copy(deep=True)
        self.E = E
        self.psi_p = psi_p
        self.psi_m = psi_m
        self.cost_series = []
        self.sample_size = len(self.E) - Setting.data_offset
        self.memory_point = []
        self.chunk = None
        self.chunk_gen = None
        self.point_gen = None

        self.verbose_step = Setting.verbose_steps
        self.computation_statistics = {}
        self.q_historical = [q_0]
        self.bench_cost = None
        self.regret = None
        self.start = Setting.data_offset
        self.end = self.start + self.sample_size

    @staticmethod
    def dot(x, q):
        return float(np.squeeze(x.reshape((1, -1)) @ q.values.reshape((-1, 1))))

    def initialize_q(self):
        # q_0 = [1e-2 for _ in range(len(Setting.feature_case))]
        # q_0 = [1e-2 for _ in self.x.columns]
        q_0 = pd.DataFrame({c: np.array([1e-2]) for c in self.x.columns})
        # q_0 = pd.DataFrame({Label.dk1da: np.array([1])})
        # q_0 = pd.DataFrame([1e-2, 1, 1e-2, 1.11, -1.65, 0.12], columns=self.x.columns)
        # if self.price_mode == Label.bidding_mode:
        #     q_0[Label.dk1mae] = 1.
        # else:
        q_0[Label.dk1da] = 1.
        return q_0

    def evaluate_true_cost(self, q):
        # q is a list

        start = self.start
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
        x, e, pp, pm = self.last_memory_point()
        e_hat = max(0, min(Setting.wind_capacity, self.dot(x, q)))
        true_cost = pp * max(0, e - e_hat) + pm * max(0, e_hat - e)
        self.cost_series.append(true_cost)

    def compute_benchmark_cost(self):
        start = self.start
        aol_series = np.array(
            [pp * max(0, e - x) + pm * max(0, x - e)
             for x, e, pp, pm in zip(self.x[Label.dk1da].values[start:], self.E[start:],
                                     self.psi_p[start:], self.psi_m[start:])])
        self.bench_cost = aol_series

        return aol_series

    def update_q(self):
        x_data, E, psi_p, psi_m = self.chunk

        y_bounds = (0, Setting.wind_capacity)
        if self.price_mode == Label.bidding_mode:
        #     y_bounds = (None, None)    # Second step <----
            pass
        elif self.price_mode == Label.forecasting_mode:
            psi_p = pd.Series(np.ones(len(psi_p)), index=psi_p.index)
            psi_m = pd.Series(np.ones(len(psi_m)), index=psi_m.index)
        q_j, obj_func = compute_optimal_lp_q(E, psi_p, psi_m, x_data, y_bounds, q_0=self.q.values[0])
        # self.q = q_j
        self.q = pd.DataFrame([q_j], columns=self.q.columns)

    def point_generator(self):
        m = 1
        # start = 100
        start = self.start
        for i in range(start + 1, self.end + 1):
            yield [(x, e, pp, pm) for x, e, pp, pm in
                              zip(self.x.values[i-m:i, :], self.E[i-m:i], self.psi_p[i-m:i], self.psi_m[i-m:i])]
        # Si start = 100 empieza en el pto 101 y coge hacia atrás los ptos necesarios i.e. m=3, (101, 100, 99).
        # i llega hasta self.sample_size -1 para acceder al último pto se necesita [self.sample_size-1: self.sample_size]
        # por eso es necesario sumar 1

    def chunk_generator(self):
        m = Setting.chunk_length
        # start = 100
        start = self.start
        for i in range(start + 1, self.end + 1, Setting.up_step):
            if Setting.single_feature:
                yield self.x[[Label.dk1da]].iloc[i-m:i, :], self.E[i-m:i], self.psi_p[i-m:i], self.psi_m[i-m:i]
            else:
                yield self.x.iloc[i-m:i, :], self.E[i-m:i], self.psi_p[i-m:i], self.psi_m[i-m:i]

        # Si start = 100 empieza en el pto 101 y coge hacia atrás los ptos necesarios i.e. m=3, (101, 100, 99).
        # i llega hasta self.sample_size -1 para acceder al último pto se necesita [self.sample_size-1: self.sample_size]
        # por eso es necesario sumar 1

    def store_step_values(self):
        # self.q_historical.append(self.q)
        # self.q_historical.append(self.q.copy(deep=True))
        self.q_historical.append(pd.DataFrame(self.q.values, columns=self.q.columns))

    def initialize_memory(self):
        self.chunk_gen = self.chunk_generator()
        self.point_gen = self.point_generator()

    def update_point(self):
        self.memory_point = next(self.point_gen)

    def update_chunk(self):
        self.chunk = next(self.chunk_gen)

    def last_memory_point(self):
        return self.memory_point[-1]

    def online_bidding(self):

        self.computation_statistics['start'] = process_time()
        self.initialize_memory()
        for i in range(self.sample_size):
            if i % self.verbose_step == 0 and i > 0:
                print('The value of q in iteration {} is {}'.format(i + 1, list(self.q.values)))
            self.update_point()
            self.evaluate_q_cost(self.q)
            if i % Setting.up_step == 0:
                print(f'Step {i}. Updating q')
                self.update_chunk()
                self.update_q()
            self.store_step_values()

        bench_cost = self.compute_benchmark_cost()
        self.cost_series = np.array(self.cost_series)

        self.computation_statistics['end'] = process_time()
        self.computation_statistics['elapsed_time'] = \
            self.computation_statistics['end'] - self.computation_statistics['start']
        self.computation_statistics['n_samples'] = self.sample_size
        self.computation_statistics['OL_cost'] = np.sum(self.cost_series) / self.sample_size
        self.computation_statistics['FO_cost'] = np.sum(bench_cost) / self.sample_size
        self.computation_statistics['final_q'] = self.q

    def print_computation_report(self):
        if len(self.computation_statistics) == 0:
            raise ValueError('No statistics recorded.')

        summary = [
            'Solver summary statistics:',
            '--------------------------',
            'Elapsed time: {}'.format(self.computation_statistics['elapsed_time']),
            'Sample size: {}'.format(self.computation_statistics['n_samples']),
            'Benchmark cost: {}'.format(self.computation_statistics['FO_cost']),
            'Avg. cost: {}'.format(self.computation_statistics['OL_cost']),
         ]

        print('\n' + '\n'.join(summary) + '\n')

    def export_results(self):
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        q_df = pd.concat(self.q_historical).reset_index(drop=True)
        start = self.start
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

        q_fixed, cost_fixed = compute_optimal_lp_q(E, psi_p, psi_m, x, (0, Setting.wind_capacity))
        E_fixed = (x * q_fixed).sum(axis=1)
        E_fixed.name = 'E_fixed'
        E_fix_cost = pd.Series(self.evaluate_true_cost(q_fixed), index=E_d.index, name='E_fix_cost')
        rounds = range(1, len(E_fix_cost) + 1)
        regret = (cost.cumsum() - E_fix_cost.cumsum()) / rounds
        regret.name = 'regret'
        self.regret = regret

        report = pd.DataFrame({'Setting': [
            'Elapsed time',
            'Sample size:',
            'Chunk len:',
            'mu:',
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
            self.sample_size,
            Setting.chunk_length,
            Setting.mu,
            self.price_mode.lower(),
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


def main():

    wind, b_data, h_data, x_data = load_data(Label, Setting)
    # if Setting.price_mode == Label.bidding_mode: # Second step <----
    #     Label.dk1da = Label.dk1mae
    nv_online = NVlp(x=x_data, E=wind, psi_p=b_data, psi_m=h_data)
    nv_online.online_bidding()
    nv_online.print_computation_report()
    nv_online.export_results()
    nv_online.plot_iteration_evolution()


if __name__ == '__main__':
    main()
