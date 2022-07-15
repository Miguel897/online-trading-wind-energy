import six
import time
import os
import sys
from pyomo import __version__ as pev
from pyomo.core.base import Var
from pyomo.environ import SolverFactory

if pev == '5.7.3':
    from pyomo.common.collections import Bunch
elif pev == '5.7.1':
    from pyutilib.misc import Bunch
else:
    raise ValueError('Pyomo version not supported.')

# Useful paths
# C:\Users\usuario\Miniconda3\envs\opt_env\Lib\site-packages\pyomo\solvers\plugins\solvers\__init__.py
# C:\Users\usuario\Miniconda3\envs\opt_env\Lib\site-packages\pyomo\opt\base\solvers.py
# C:\Users\usuario\Miniconda3\envs\opt_env\Lib\site-packages\pyomo\solvers\plugins\solvers\cplex_direct.py
# C:\Users\usuario\Miniconda3\envs\opt_env\Lib\site-packages\pyomo\solvers\plugins\solvers\direct_solver.py
# C:\Users\usuario\Miniconda3\envs\opt_env\Lib\site-packages\pyomo\solvers\plugins\solvers\direct_or_persistent_solver.py

# Key links:
# https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
# https://stackoverflow.com/questions/51608631/how-to-create-a-logfile-when-calling-cplex-from-python
# https://pyomo.readthedocs.io/en/stable/_modules/pyomo/solvers/plugins/solvers/cplex_direct.html
# https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.studio.help/CPLEX/ReleaseNotes/topics/releasenotes12100/removed.html
# https://stackoverflow.com/questions/43543853/advantages-of-cplex-in-pyomo-versus-cplex-in-python


class DegreeError(ValueError):
    pass


def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


# Al indicar SolverFactory('cplex_direct')  pyomo devuelve un objeto de clase
# CPLEXDirec ubicado en pyomo.solvers.plugins.solvers.cplex_direct
# Configure Python binding:
# https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html
# Logfile = False debido a
# # C:\Users\usuario\Miniconda3\envs\opt_env\Lib\site-packages\pyomo\solvers\plugins\solvers\direct_or_persistent_solver.py
# Lines 161-162
#        if self._log_file is None:
#             self._log_file = TempfileManager.create_tempfile(suffix='.log')


def custom_solver_factory(solver_name, *args, **kwargs):
    """Esta funcion sobreescribe la funcion _apply_solver de CPLEXDirect con una nueva definicion.
    Ademas es necesario indicar un valor para la keyword logfile distinto de None, ej:
    opt.solve(model, symbolic_solver_labels=False, tee=False, keepfiles=True, logfile=False)
    Con esta convinacion deben desaparecer por completo los logfiles."""
    optimizer = SolverFactory(solver_name, *args, **kwargs)
    optimizer._apply_solver = apply_solver_custom.__get__(optimizer)

    return optimizer


def apply_solver_custom(self):
    if not self._save_results:
        for block in self._pyomo_model.block_data_objects(descend_into=True, active=True):
            for var in block.component_data_objects(ctype=Var, descend_into=False, active=True, sort=False):
                var.stale = True
    # In recent versions of CPLEX it is helpful to manually open the
    # log file and then explicitly close it after CPLEX is finished.
    # This ensures that the file is closed (and unlocked) on Windows
    # before the TempfileManager (or user) attempts to delete the
    # log file.  Passing in an opened file object is supported at
    # least as far back as CPLEX 12.5.1 [the oldest version
    # supported by IBM as of 1 Oct 2020]
    _any_valid_logfile = isinstance(self._log_file, six.string_types)
    if _any_valid_logfile:
        _log_file = (open(self._log_file, 'a'),)

        if self._tee:
            def _process_stream(arg):
                sys.stdout.write(arg)
                return arg

            _log_file += (_process_stream,)

    try:

        if _any_valid_logfile:
            self._solver_model.set_results_stream(*_log_file)
        elif self._tee:
            pass
        else:
            self._solver_model.set_results_stream(None)
            # self._solver_model.set_results_stream(open(os.devnull, "w"))

            # https://stackoverflow.com/questions/51608631/how-to-create-a-logfile-when-calling-cplex-from-python
            # prob.set_log_stream(None)
            # prob.set_error_stream(None)
            # prob.set_warning_stream(None)
            # prob.set_results_stream(None)

            # with cplex.Cplex() as cpx, \
            #     open("cplex.log") as cplexlog:
            #     cpx.set_results_stream(cplexlog)
            #     cpx.set_warning_stream(cplexlog)
            #     cpx.set_error_stream(cplexlog)
            #     cpx.set_log_stream(cplexlog)

        obj_degree = self._objective.expr.polynomial_degree()
        if obj_degree is None or obj_degree > 2:
            raise DegreeError('CPLEXDirect does not support expressions of degree {0}.' \
                              .format(obj_degree))
        elif obj_degree == 2:
            quadratic_objective = True
        else:
            quadratic_objective = False

        num_integer_vars = self._solver_model.variables.get_num_integer()
        num_binary_vars = self._solver_model.variables.get_num_binary()
        num_sos = self._solver_model.SOS.get_num()

        if self._solver_model.quadratic_constraints.get_num() != 0:
            quadratic_cons = True
        else:
            quadratic_cons = False

        if (num_integer_vars + num_binary_vars + num_sos) > 0:
            integer = True
        else:
            integer = False

        if integer:
            if quadratic_cons:
                self._solver_model.set_problem_type(self._solver_model.problem_type.MIQCP)
            elif quadratic_objective:
                self._solver_model.set_problem_type(self._solver_model.problem_type.MIQP)
            else:
                self._solver_model.set_problem_type(self._solver_model.problem_type.MILP)
        else:
            if quadratic_cons:
                self._solver_model.set_problem_type(self._solver_model.problem_type.QCP)
            elif quadratic_objective:
                self._solver_model.set_problem_type(self._solver_model.problem_type.QP)
            else:
                self._solver_model.set_problem_type(self._solver_model.problem_type.LP)

        # if the user specifies a 'mipgap'
        # set cplex's mip.tolerances.mipgap
        if self.options.mipgap is not None:
            self._solver_model.parameters.mip.tolerances.mipgap.set(float(self.options.mipgap))

        for key, option in self.options.items():
            if key == 'mipgap':  # handled above
                continue
            opt_cmd = self._solver_model.parameters
            key_pieces = key.split('_')
            for key_piece in key_pieces:
                opt_cmd = getattr(opt_cmd, key_piece)
            # When options come from the pyomo command, all
            # values are string types, so we try to cast
            # them to a numeric value in the event that
            # setting the parameter fails.
            try:
                opt_cmd.set(option)
            except self._cplex.exceptions.CplexError:
                # we place the exception handling for
                # checking the cast of option to a float in
                # another function so that we can simply
                # call raise here instead of except
                # TypeError as e / raise e, because the
                # latter does not preserve the Cplex stack
                # trace
                if not _is_numeric(option):
                    raise
                opt_cmd.set(float(option))

        t0 = time.time()
        self._solver_model.solve()
        t1 = time.time()
        self._wallclock_time = t1 - t0

    finally:
        if _any_valid_logfile:
            self._solver_model.set_results_stream(None)
            _log_file[0].close()

    return Bunch(rc=None, log=None)
