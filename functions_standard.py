import pandas as pd
import numpy as np
import time
import datetime
import os
import cProfile
import pstats
import functools
from time import process_time
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns
from itertools import product


# STANDARD SHORTCUTS

CSV = ".csv"
EXCEL = ".xlsx"
PNG = ".png"
TXT = ".txt"


# STANDARD FUNCTIONS


class PlatformContext:

    def __init__(self):
        from platform import system, node
        self._windows: str = 'Windows'
        self._linux: str = 'Linux'
        self._picasso: str = 'picasso'
        self._ada: str = 'pythonsimulator'
        self._pc: str = 'DESKTOP-8Q9B0B0'
        self.system: str = system()
        self.name: str = node()
        self.is_windows: bool = True if self.system == self._windows else False
        self.is_linux: bool = True if self.system == self._linux else False
        self.is_ada: bool = True if self.name == self._ada else False
        self.is_pc: bool = True if self.name == self._pc else False
        if (self.name == self._picasso) or not (self.is_pc or self.is_ada):
            self.is_picasso: bool = True
        else:
            self.is_picasso: bool = False

    def update_environ_variables(self, threads=None, temporary_dir=None, neos=False, default_slurm_id=1):
        from os import environ, getcwd

        if neos:
            print('Adding default neos email.')
            environ['NEOS_EMAIL'] = 'xxx@gmail.com'

        if threads is not None:
            print(f'Setting max no. of threads to {threads}.')
            environ['OMP_NUM_THREADS'] = str(threads)

        temporary_dir = getcwd() if temporary_dir == 'cwd' else temporary_dir
        if temporary_dir is not None:
            print(f'Updating temporary dir to: {temporary_dir}')
            if self.is_linux:
                environ['TMPDIR'] = temporary_dir
            if self.is_windows:
                environ['temp'] = temporary_dir

        if not self.is_picasso:
            print(f'Setting SLURM_ARRAY_TASK_ID to {default_slurm_id}')
            environ['SLURM_ARRAY_TASK_ID'] = str(default_slurm_id)
        else:
            try:
                environ['SLURM_ARRAY_TASK_ID']
            except KeyError:
                print(f'Setting SLURM_ARRAY_TASK_ID to {default_slurm_id}')
                environ['SLURM_ARRAY_TASK_ID'] = str(default_slurm_id)


class SimulationContext:
    def __init__(self, default_simulation_mode, full_simulation_list, batch_size=1, sim_cases_indexes=None):

        self.simulation_mode = default_simulation_mode
        self.full_simulation_list = full_simulation_list
        self.batch_size = batch_size
        self.sim_cases_indexes = sim_cases_indexes
        self.script_simulation_list = None

        option_mode = self.get_script_arguments(key='mode')
        if option_mode is not None:
            self.simulation_mode = option_mode
        self.compute_script_simulation_list()

    @staticmethod
    def get_script_arguments(key=None):
        from sys import argv
        try:
            arguments = eval('{' + argv[1] + '}')
            if key is None:
                return arguments
            else:
                return arguments.pop(key, None)
        except IndexError:
            return None

    def compute_script_simulation_list(self):
        from os import environ
        if self.simulation_mode == 'all':
            self.script_simulation_list = self.full_simulation_list
        if self.simulation_mode == 'batch':
            array_id = int(environ['SLURM_ARRAY_TASK_ID']) - 1
            self.script_simulation_list = [self.full_simulation_list[i] for i in
                                           range(array_id * self.batch_size, (array_id + 1) * self.batch_size)]
        elif self.simulation_mode == 'index':
            self.script_simulation_list = [self.full_simulation_list[i] for i in self.sim_cases_indexes]


def float_dict_to_string(float_dict, decimal=3):
    """Change the format of a dict with a lot of zeros.
    """
    string = "{0:." + str(decimal) + "f}"
    target = str(string.format(0))
    element_list = [element[0].replace("'", "") + (": " + string + ", ").format(
        element[1]) for element in float_dict.items()]
    element_list = [element.replace(target, "0") for element in element_list]
    element_list = ''.join(element for element in element_list)

    return element_list[:-2]


def append_ones(data, df_output=False, column_names=None):
    """Append a column of ones into a 2D array or DataFrame.
    """
    if isinstance(data, np.ndarray):
        shape = np.shape(data)
        if len(shape) != 2:
            raise ValueError('Invalid array dimension.')

        data = np.column_stack([np.ones((np.shape(data)[0], 1)), data])
        if df_output:
            if column_names is None:
                column_names = [f'c{i}' for i in range(shape[1])]
            data = pd.DataFrame(data, columns=['ones'] + column_names)
    elif isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        data = data.copy()
        data['ones'] = 1
        data = data[['ones'] + columns]
    else:
        raise ValueError('Invalid object.')

    return data


def data_to_dict(data, start=0):
    """Transform array-like data into a dictionary where the key is a tuple with the position of the element.
    """

    if isinstance(data, np.ndarray):
        shp = np.shape(data)
        length = len(shp)
        if length < 1 or length > 2:
            raise ValueError('Invalid array or object.')

        if length == 2:
            dictionary = {(i + start, j + start): data[i, j] for (i, j) in product(range(shp[0]), range(shp[1]))}
        else:
            dictionary = {i + start: data[i] for i in range(shp[0])}

    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        shp = np.shape(data)
        length = len(shp)
        if length < 1 or length > 2:
            raise ValueError('Invalid array or object.')

        if length == 2:
            dictionary = {(i, j): data.at[i, j] for (i, j) in product(data.index, data.columns)}
        else:
            dictionary = {i: data[i] for i in data.index}
    else:
        raise ValueError('Invalid object.')

    return dictionary


def merge_same_key_dictionaries(dictionary_list, base_dict=None, df_output=False, key_error=True):
    """Merge several dictionaries with the same keys into one of the form {key: list}.
    The output can be a df.
    """

    if base_dict is None:
        base_dict = {}

    all_keys = []
    for dictionary in dictionary_list:
        all_keys.extend(list(dictionary.keys()))
    all_keys = set(all_keys)

    for key in all_keys:
        base_dict[key] = []
        for dictionary in dictionary_list:
            try:
                base_dict[key].append(dictionary[key])
            except KeyError as error:
                if key_error:
                    raise KeyError(str(error))
                else:
                    pass

    if df_output:
        return pd.DataFrame(base_dict)
    else:
        return base_dict


def get_timestamp_label(underscore=False):
    """Returns a string timestamp label of the form YYYYMMDD_HHMMSS.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if underscore:
        timestamp += '_'
    return timestamp


def create_directory(folder_name="", parent_path=None, parents=True, exist_ok=True, timestamp=False):
    """Handy function for creating directories.
    """
    if not parent_path:
        parent_path = os.getcwd()
    prefix = get_timestamp_label(underscore=True) if timestamp else ''
    full_path = os.path.join(parent_path, prefix + folder_name)
    Path(full_path).mkdir(parents=parents, exist_ok=exist_ok)

    return full_path


def save_code(source_files, simulation_path, folder_name='Source'):
    """Copy source files of simulation to the results folder.
    """
    create_directory(folder_name=folder_name, parent_path=simulation_path)
    from_path = os.getcwd()
    to_path = os.path.join(simulation_path, folder_name)
    for file in source_files:
        copy2(os.path.join(from_path, file), to_path)


def save_code2(simulation_path, folder_name='Source', ignore=None, ignore_default=True):
    """Copy source files of simulation to the results folder.
    """

    from_path = os.getcwd()
    to_path = os.path.join(simulation_path, folder_name)

    ignore_default_patterns = ('.git', '.gitignore', '.idea', '__pycache__', '*.log', '*.sh', '*.lnk')
    if ignore_default and ignore is not None:
        ignore = ignore_default_patterns + ignore
    elif ignore_default:
        ignore = ignore_default_patterns

    if ignore is None:
        copytree(from_path, to_path)
    else:
        copytree(from_path, to_path, ignore=ignore_patterns(*ignore))


def time_it(func):
    def wrapper(*args, **kwargs):
        start = process_time()
        results = func(*args, **kwargs)
        print(f'Elapsed time: {process_time() - start}')
        return results
    return wrapper


def profiler(criteria='cumtime', subcalls=True, builtins=False):
    def profiler_dec(func):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile(subcalls=subcalls, builtins=builtins)
            pr.enable()
            value = func(*args, **kwargs)
            pr.disable()
            # pr.print_stats(sort=criteria)
            st = pstats.Stats(pr)
            st.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats('functions_core')
            return value
        return wrapper
    return profiler_dec


def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def list_to_df(data, columns):

    len_columns = len(columns)
    container = {f'key{i}': [] for i in range(len_columns)}
    for pos, val in enumerate(data):
        i = divmod(pos, len_columns)[1]
        container[f'key{i}'].append(val)
    df = pd.DataFrame(container).rename(dict(zip(list(container.keys()), columns)))

    return df


def feature_scaling(scal_type, *X_mats):
    """Perform scaling in feature matrices
    """
    assert len(X_mats) >= 2
    assert scal_type in ["max", "mnmx", "stdr"]

    X_mats = list(X_mats)  # Argument may be tuple
    if type(X_mats[0]) == np.ndarray:
        if scal_type == "max":
            max_tr_val = X_mats[0].max(axis=0)
            for i, x_m in enumerate(X_mats):
                X_mats[i] = x_m / max_tr_val
        elif scal_type == "mnmx":
            min_tr_val = X_mats[0].min(axis=0)
            max_tr_val = X_mats[0].max(axis=0)
            for i, x_m in enumerate(X_mats):
                X_mats[i] = (x_m - min_tr_val) / (max_tr_val - min_tr_val)
        elif scal_type == "stdr":
            mean_tr_val = X_mats[0].mean(axis=0)
            std_tr_val = X_mats[0].std(axis=0)
            for i, x_m in enumerate(X_mats):
                X_mats[i] = (x_m - mean_tr_val) / std_tr_val
        for i, x_m in enumerate(X_mats):
            X_mats[i][np.isnan(x_m)] = 0

    elif type(X_mats[0]) == pd.DataFrameType:
        if scal_type == "max":
            max_tr_val = X_mats[0].max()
            for i, x_m in enumerate(X_mats):
                X_mats[i] = x_m / max_tr_val
        elif scal_type == "mnmx":
            min_tr_val = X_mats[0].min()
            max_tr_val = X_mats[0].max()
            for i, x_m in enumerate(X_mats):
                X_mats[i] = (x_m - min_tr_val) / (max_tr_val - min_tr_val)
        elif scal_type == "stdr":
            mean_tr_val = X_mats[0].mean()
            std_tr_val = X_mats[0].std()
            for i, x_m in enumerate(X_mats):
                X_mats[i] = (x_m - mean_tr_val) / std_tr_val
        for i, x_m in enumerate(X_mats):
            X_mats[i] = x_m.fillna(0, inplace=False)

    return X_mats


def check_limits(series, bounds, verbose=False, tolerance=1e-6):
    """Check a data series is between bounds. Modify elements outside limits and count them.
    """
    c_lower, c_upper, outliers = 0, 0, []
    tolerance = (np.max(series) - np.min(series)) * tolerance
    for i in range(len(series)):
        if series[i] < bounds[0]:
            if series[i] < (bounds[0] - tolerance):
                outliers.append(series[i])
                series[i] = bounds[0]
                c_lower += 1
            else:
                series[i] = bounds[0]
        elif series[i] > bounds[1]:
            if series[i] > (bounds[1] + tolerance):
                outliers.append(series[i])
                series[i] = bounds[1]
                c_upper += 1
            else:
                series[i] = bounds[1]

    p_lower = c_lower / len(series) * 100
    p_upper = c_upper / len(series) * 100
    if (c_lower > 0) or (c_upper > 0):
        if verbose:
            print("There were {:.2f} / {:.2f} of points below/above limit.".format(p_lower, p_upper))

    return series, {"lb_breaks": p_lower, "ub_breaks": p_upper, "outliers": outliers}


def natural_cubic_spline_basis(x, df, ones=True, cname=None):

    x = np.squeeze(x)

    def pcubic(u):
        return np.maximum(u, 0.)**3

    def dk(x, kk):
        return (pcubic(x - knots[kk-1]) - pcubic(x - knots[-1])) / (knots[-1] - knots[kk-1])

    N = len(x)
    percentiles = np.linspace(0, 100, df + 2)[1:-1]
    knots = np.percentile(x, percentiles, interpolation='lower')

    basis = {}
    for i in range(1, df + 1):
        if i == 1:
            basis[i] = np.ones(N)
        elif i == 2:
            basis[i] = x
        else:
            k = i - 2
            basis[i] = dk(x, k) - dk(x, df - 1)
    if not ones:
        basis.pop(1)
    if cname is not None:
        basis = {cname + f'_b{k}': v for (k, v) in basis.items()}

    return pd.DataFrame(basis)


def augment_feature_space(df, columns, dgf):

    df = df.copy(deep=True)
    for col in columns:
        col_ag = natural_cubic_spline_basis(df[col].values, dgf, ones=False, cname=col)
        col_ag.index = df.index
        col_loc = df.columns.get_loc(col)
        df.drop(columns=col, inplace=True)
        df = pd.concat([df.iloc[:, :col_loc], col_ag, df.iloc[:, col_loc:]], axis=1)

    return df


def half_space_projection(y, a, b):

    if np.dot(a, y) <= b:
        return y
    else:
        return y - (np.dot(a, y) - b) / np.linalg.norm(a, ord=2)**2 * a


def parallel_half_space_projection(y, a, lh, rh):
    # Check https://math.stackexchange.com/questions/318740/orthogonal-projection-onto-a-half-space
    # Finds the projection x of vector y onto the intersection of half spaces of the form lh <= a'x <= rh
    # Half spaces are convex and the intersection is also convex. The solution in unique.

    if lh <= np.dot(a, y) <= rh:
        return y
    elif np.dot(a, y) > rh:
        return half_space_projection(y, a, rh)
    elif np.dot(a, y) < lh:
        return half_space_projection(y, -a, -lh)


if __name__ == '__main__':
    pass

