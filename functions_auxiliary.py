from math import exp, log
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
    explained_variance_score


def sigmoidp(x):
    if x <= 0:
        return 1 / (1 + exp(x))
    else:
        return 1 - 1 / (1 + exp(-x))


def sigmoidn(x):
    if x <= 0:
        return 1 - 1/(1 + exp(x))
    else:
        return 1/(1 + exp(-x))


def exp_func_01(x):
    return log(1 + exp(-x))


def exp_func_02(x):
    if x <= 0:
        return exp(x) / (1 + exp(x)) ** 2
    else:
        return 1 / (1 + exp(-x)) - 1 / (1 + exp(-x)) ** 2


def exp_func_03(x, alpha, capacity):
    x = max(- capacity, min(capacity, x)) / alpha
    if x <= 0:
        return exp(x) / (1 + exp(x)) ** 2
    else:
        return 1 / (1 + exp(-x)) - 1 / (1 + exp(-x)) ** 2


def check_q_j(coef_list, labels, err_limit=0.05):
    """Supervise feature regressors to detect Nans.
    """
    assert not any([isinstance(x, (int, float)) for x in labels])
    assert len(coef_list) == len(labels)

    err = sum([1. if q is None else 0 for q in coef_list]) / len(coef_list)
    coef_dict = {key: value for (key, value) in zip(labels, coef_list)}
    if err > err_limit:
        print('Coeficients: ', coef_dict)
        raise ValueError(
            'The solver is unable to find a solution for a significant amount of columns. Error rate={0}'.format(err))
    elif err > 0:
        print('Coeficients: ', coef_dict)
        print('The solver is unable to find a solution for some columns. Error rate={0}'.format(err))

    return coef_dict, err


def compute_metrics(y_true, y_pred, b, h, metrics=None):
    """Evaluates different metrics.
    """
    metrics_dict = {}
    if 'MAE' in metrics:
        metrics_dict['MAE'] = mean_absolute_error(y_true, y_pred)
    if 'MAPE' in metrics:
        metrics_dict['MAPE'] = np.mean(np.abs(y_true - y_pred) / y_true) * 100
    if 'RMSE' in metrics:
        metrics_dict['RMSE'] = mean_squared_error(y_true, y_pred) ** 0.5
    if 'R2' in metrics:
        metrics_dict['R2'] = r2_score(y_true, y_pred)
    if 'EVAR' in metrics:
        metrics_dict['EVAR'] = explained_variance_score(y_true, y_pred)
    if 'AOL' in metrics:
        y = list(y_true - y_pred)
        try:
            h[0]
        except TypeError:
            h, b = h * np.ones(len(y)), b * np.ones(len(y))
        metrics_dict['AOL'] = sum(
            [h[i] * abs(y[i]) if y[i] < 0 else b[i] * y[i] for i in range(len(y))]
        ) / len(y)

    return metrics_dict


def points_to_equation_system(line_list, test_point):
    """
    """

    test_point = test_point.reshape((2, 1))
    dimension = len(line_list['from'].iat[0])
    A = np.zeros((0, dimension))
    b = np.zeros((0, 1))
    line_index = list(line_list.index)
    for line in line_index:
        A_row, b_row = compute_line_coefficients(
            line_list.at[line, 'from'], line_list.at[line, 'to']
        )
        if A_row @ test_point <= b_row:
            A = np.vstack((A, A_row))
            b = np.vstack((b, b_row))
        else:
            A = np.vstack((A, -1 * A_row))
            b = np.vstack((b, -1 * b_row))
    return A, b


def compute_line_coefficients(point1, point2):

    if len(point1) != len(point2):
        raise ValueError('Point dimension mismatch.')

    if point1[0] == point2[0]:
        A_row = np.array([1, 0])
        b_row = np.array(point1[0])
    elif point1[1] == point2[1]:
        A_row = np.array([0, 1])
        b_row = np.array(point1[1])
    else:
        x_coords, y_coords = zip(point1, point2)
        coef = np.polyfit(x_coords, y_coords, 1)
        # coef = np.linalg.lstsq(x_coords, y_coords, 1)
        m, c = np.round(coef, 7)
        A_row = np.array([-m, 1])
        b_row = np.array(c)
        # print(point1, point2)
        # print(f'y = {m} x + {c}')
    A_row = A_row.reshape((1, len(point1)))
    b_row = b_row.reshape((1, 1))

    return A_row, b_row


class ExponentialMovingAverage:
    def __init__(self, raw_series, lamb):
        self.raw_series = raw_series
        self.lamb = lamb
        self.ema_series = []
        self.w_t = []
        self.option = '1'
        # self.option = '2'

    def compute_smooth_series(self):
        st = self.raw_series[0]
        self.ema_series.append(st)
        for point in self.raw_series[1:]:
            if self.option == '1':
                st = (1 - self.lamb) * point + self.lamb * st
            else:
                st = self.lamb * point + (1 - self.lamb) * st
            self.ema_series.append(st)

    def compute_sum(self):
        s_t = []
        for i in range(1, len(self.raw_series) + 1):
            w_i = self.compute_weights(i)
            print(w_i, sum(w_i))

            # for k in range(1,  i + 1):
            s_t.append(sum([w * y for w, y in zip(w_i, self.raw_series[:i])]))
        return s_t

    def compute_weights(self, T):
        if self.option == '1':
            series = [(1 - self.lamb) * self.lamb ** (T-t) for t in range(1, T + 1)]
            series[0] = self.lamb ** (T - 1)
        else:
            series = [(1 - self.lamb) ** (T-t) * self.lamb for t in range(1, T + 1)]
            series[0] = (1 - self.lamb) ** (T - 1)

        return series

    def plot_series(self):
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.raw_series)), self.raw_series, color='cyan', label='raw')
        plt.plot(range(len(self.ema_series)), self.ema_series, color='orange', label='ema')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # ema = ExponentialMovingAverage([4, 7, 2, 6, 8], 0.7)
    ema = ExponentialMovingAverage([4, 7, 2, 6, 8, 9, 3], 0.9)
    ema.compute_smooth_series()
    print(ema.ema_series)
    print(ema.compute_sum())
    ema.plot_series()

