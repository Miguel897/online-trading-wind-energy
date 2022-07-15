import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functions_standard import time_it
from functions_auxiliary import sigmoidp, exp_func_01, exp_func_02


class NVGradient:

    steepest_method = 'steepest'
    newton_method = 'newton'
    valid_methods = ['steepest', 'newton']

    def __init__(self, q_0, x, E, psi_p, psi_m, alpha, tolerance, iteration_limit,
                 eta=1., verbose_step=10, record_q_historical=True):
        self.q = q_0
        self.x = x
        self.E = E
        self.psi_p = psi_p
        self.psi_m = psi_m
        self.alpha = alpha
        self.eta = eta
        self.tolerance = tolerance
        self.iteration_limit = iteration_limit
        self.gradient = np.NAN
        self.hessian = np.NAN
        self.solve_method = None
        self.sample_size = len(self.E)
        self.verbose_step = verbose_step
        self.computation_statistics = {}
        self.q_historical = [q_0.values]
        self.record_q_historical = record_q_historical

    @staticmethod
    def dot(x, q):
        return float(np.squeeze(x.reshape((1, -1)) @ q.values.reshape((-1, 1))))

    def evaluate_true_objective_function(self, q=None):
        if q is None:
            q = self.q
        return 1 / self.sample_size * np.sum(np.array(
            [pp * max(0, e - self.dot(x, q)) + pm * max(0, self.dot(x, q) - e)
             for x, e, pp, pm in zip(self.x.values, self.E, self.psi_p, self.psi_m)]))

    def evaluate_smooth_objective_function(self, q):
        return 1 / self.sample_size * np.sum(np.array(
            [pp * (e - self.dot(x, q)) + self.alpha * (pp + pm) * exp_func_01((e - self.dot(x, q)) / self.alpha)
             for x, e, pp, pm in zip(self.x.values, self.E, self.psi_p, self.psi_m)]))

    def compute_hessian_matrix(self, q, update_hessian=True):

        c_vec = 1 / self.sample_size * np.array([(pp + pm) / self.alpha * exp_func_02((e - self.dot(x, q)) / self.alpha)
                                               for x, e, pp, pm in zip(self.x.values, self.E, self.psi_p, self.psi_m)])

        hessian_t = [c_t * x_t.reshape((-1, 1)) @ x_t.reshape((1, -1)) for x_t, c_t in zip(self.x.values, c_vec)]
        hessian = np.array(hessian_t).sum(axis=0)

        if update_hessian:
            self.hessian = hessian

        return hessian

    def compute_gradient(self, q, method='newton', update_gradient=True):

        if method == self.steepest_method:
            aux = 1 / self.sample_size * np.array([-pp + (pp + pm) * sigmoidp((e - self.dot(x, q)) / self.alpha)
                            for x, e, pp, pm in zip(self.x.values, self.E, self.psi_p, self.psi_m)]).reshape((-1, 1))
            gradient = (aux * self.x).sum()

        elif method == self.newton_method:
            self.compute_gradient(q, method=self.steepest_method, update_gradient=True)

            try:
                self.compute_hessian_matrix(q, update_hessian=True)
                gradient = np.linalg.inv(self.hessian) @ self.gradient
                gradient = pd.DataFrame([gradient], columns=self.gradient.index)
            except np.linalg.LinAlgError:
                gradient, update_gradient = self.gradient, False

        if update_gradient:
            self.gradient = gradient

        return gradient

    def update_q(self):
        if self.solve_method == self.steepest_method:
            self.q -= self.eta * self.gradient
        if self.solve_method == self.newton_method:
            self.q -= self.gradient

    def compute_optimal_q(self, method):

        if method not in self.valid_methods:
            raise ValueError('Invalid method.')

        self.solve_method = method
        self.computation_statistics['start'] = time.process_time()
        for i in range(self.iteration_limit):
            if i % self.verbose_step == 0 and i > 0:
                print('The value of q in iteration {} is {}'.format(i + 1, list(self.q.values)))
                print('Gradient is: {}'.format(list(self.gradient.values)))
            self.compute_gradient(self.q, method=self.solve_method, update_gradient=True)
            # if all(abs(self.gradient) < self.tolerance):
            if (abs(self.gradient) < self.tolerance).values.all():
                break
            else:
                self.update_q()
                if self.record_q_historical:
                    # self.q_historical.append(self.q.copy(deep=True))
                    self.q_historical.append(self.q.values)

        self.computation_statistics['end'] = time.process_time()
        self.computation_statistics['elapsed_time'] = \
            self.computation_statistics['end'] - self.computation_statistics['start']
        self.computation_statistics['iterations'] = i + 1
        self.computation_statistics['iteration_limit'] = self.iteration_limit
        self.computation_statistics['eta'] = self.eta
        self.computation_statistics['alpha'] = self.alpha
        self.computation_statistics['tolerance'] = self.tolerance
        self.computation_statistics['solver_method'] = self.solve_method
        self.computation_statistics['gradient'] = self.gradient
        self.computation_statistics['solution'] = self.q

        return self.q, self.computation_statistics

    def print_computation_report(self):
        if len(self.computation_statistics) == 0:
            raise ValueError('No statistics recorded.')

        summary = [
            'Solver summary statistics:',
            '--------------------------',
            'Starting time: {}'.format(self.computation_statistics['start']),
            'Ending time: {}'.format(self.computation_statistics['end']),
            'Elapsed time: {}'.format(self.computation_statistics['elapsed_time']),
            'Solver method: {}'.format(self.computation_statistics['solver_method']),
            'Iterations: {} out of {} max'.format(
                self.computation_statistics['iterations'], self.computation_statistics['iteration_limit']),
            'Tolerance: {}'.format(self.computation_statistics['tolerance']),
            'alpha parameter: {}'.format(self.computation_statistics['alpha']),
            'Gradient value:\n {}'.format(self.computation_statistics['gradient'].to_string(index=False)),
            'Current solution:\n {}'.format(self.computation_statistics['solution'].to_string(index=False))
         ]

        if self.solve_method == NVGradient.steepest_method:
            summary.append(
                'Eta parameter: {}'.format(self.computation_statistics['eta']),
            )

        print('\n' + '\n'.join(summary) + '\n')

    def plot_iteration_evolution(self):
        # print(self.q_historical)
        q_vector = [pd.DataFrame(q) for q in self.q_historical]
        true_objective = [self.evaluate_true_objective_function(q) for q in q_vector]
        smooth_objective = [self.evaluate_smooth_objective_function(q) for q in q_vector]
        iterations = range(1, len(q_vector) + 1)

        plt.plot(iterations, true_objective, color='blue', label='True obj.')
        plt.plot(iterations, smooth_objective, color='cyan', label='Smooth obj.')
        plt.xticks(np.arange(min(iterations) - 1, max(iterations) + 1, 1.0))
        plt.xlabel('Iterations')
        plt.ylabel('Obj. value')
        plt.title(r'Convergence for $\alpha = $' + str(self.alpha))
        plt.legend()
        plt.show()
        print(true_objective)
        print(smooth_objective)
        print([smooth_objective[-1] < x for x in smooth_objective])
        print([true_objective[-1] < x for x in true_objective])


@dataclass
class Setting:
    n_samples: int = 100
    error: float = 7.
    gen_capacity: float = 100
    solver_method: str = NVGradient.newton_method
    # iteration_limit: int = int(5 * 1e5)
    iteration_limit: int = int(1 * 1e3)
    eta: float = 1e-3
    alpha: float = 0.05
    # alpha: float = 0.02
    tolerance: float = 1e-2
    verbose_steps: int = 50
    # verbose_steps: int = 1


@time_it
def routine01(q_0, x, E, b_i, h_i):
    # Solves the problem once
    gradient_nv = NVGradient(q_0=q_0, x=x, E=E, psi_p=b_i, psi_m=h_i,
                             alpha=Setting.alpha, eta=Setting.eta, tolerance=Setting.tolerance,
                             iteration_limit=Setting.iteration_limit, verbose_step=Setting.verbose_steps)

    gradient_nv.compute_optimal_q(method=Setting.solver_method)
    gradient_nv.print_computation_report()
    gradient_nv.plot_iteration_evolution()


@time_it
def routine02(q_0, x, E, b_i, h_i):
    # Iteratively solves for decreasing alpha reusing the last solution as initial value.
    gradient_nv = NVGradient(q_0=q_0, x=x, E=E, psi_p=b_i, psi_m=h_i,
                             alpha=Setting.alpha, eta=Setting.eta, tolerance=Setting.tolerance,
                             iteration_limit=Setting.iteration_limit, verbose_step=Setting.verbose_steps)


    for alpha in [1., 0.05, 0.02]:
        gradient_nv.alpha = alpha
        gradient_nv.compute_optimal_q(method=Setting.solver_method)
        gradient_nv.print_computation_report()


def newton_solver(q_0, data):

    x_ij, y_i = data['x_ij'], data['y_i']
    b_i, h_i = data['b_i'], data['h_i']
    y_bounds = data['y_bounds']
    N, p = np.shape(x_ij)

    gradient_nv = NVGradient(q_0=q_0, x=x_ij, E=y_i, psi_p=b_i, psi_m=h_i,
                             alpha=Setting.alpha, eta=Setting.eta, tolerance=Setting.tolerance,
                             iteration_limit=Setting.iteration_limit, verbose_step=Setting.verbose_steps)

    for alpha in [1., 0.05, 0.02]:
        gradient_nv.alpha = alpha
        gradient_nv.compute_optimal_q(method=Setting.solver_method)

    optimal_q, obj = gradient_nv.q, gradient_nv.evaluate_true_objective_function()

    return gradient_nv.q, obj


if __name__ == '__main__':
    # Data Generation

    np.random.seed(17)
    E = np.random.uniform(10, 90, Setting.n_samples)
    x = E + np.random.uniform(- Setting.error, Setting.error, Setting.n_samples)
    x = pd.DataFrame(x, columns=['E_est'])
    x['ones'] = 1
    x = x[['ones', 'E_est']]
    b_i = 4 * np.ones(Setting.n_samples)
    h_i = 1 * np.ones(Setting.n_samples)

    # Initial Solution

    q_0 = pd.DataFrame(np.array([[0.01, 1]]), columns=['ones', 'E_est'])
    # q_0 = pd.DataFrame(np.array([[0, 1]]), columns=['ones', 'E_est'])  # Through error
    # q_0 = pd.DataFrame(np.array([[3.7074, 1.0044]]), columns=['ones', 'E_est'])
    # q_0 = pd.DataFrame(np.array([[1]]), columns=['E_est'])

    # routine01(q_0, x, E, b_i, h_i)
    # routine02(q_0, x, E, b_i, h_i)


    x = E + np.random.uniform(- Setting.error, Setting.error, Setting.n_samples)
    x = pd.DataFrame(x, columns=['E_est'])
    q_n0 = pd.DataFrame({'E_est': np.array([1])})
    data = {
        'y_i': E,
        'x_ij': x,
        'b_i': b_i,
        'h_i': h_i,
        'y_bounds': (0, Setting.gen_capacity),
    }

    # q_n0 = pd.DataFrame(np.array([[0.01, 1]]), columns=['ones', 'E_est'])

    q, obj = newton_solver(q_n0, data)
    print(q, obj)