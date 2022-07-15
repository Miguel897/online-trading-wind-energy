from os.path import join
import pyomo.environ as pe
from numpy import shape, ones
from pyomo.environ import SolverFactory, SolverManagerFactory
from pyomo.opt.results.solver import SolverStatus as SolSt, TerminationCondition as TermCond
from auxiliary.optimization_utils import solve_optimization_model_direct

# Example of a solving configuration used in solve_optimization_model.
standard_solving_configuration = {
    'solver_name': 'cplex',
    'solve_options': {},
    'solver_factory_options': {'options': {
        'logfile': 'cplex.log',
        'output_clonelog': -1,
    }},
    'verbose': False,
    'display_info': False,
    'write_info': False,
    'solver_file_name': 'solver_info',
    'model_file_name': 'pyomo_model',
    'file_extension': '.txt',
    'saving_path': '',
    'timestamp': '',
}


def create_bigadata_nv_model(data):

    x_ij, y_i = data['x_ij'], data['y_i']
    b_i, h_i = data['b_i'], data['h_i']
    y_bounds = data['y_bounds']
    q_j_bounds = data['q_j_bounds']
    N, p = shape(x_ij)
    try:
        q_0 = data['q_0']
        q_0 = {k: v for k, v in enumerate(q_0)}
    except KeyError:
        q_0 = None
        # pass
    try:
        b_i[0]  # Chek if vector-like or integer
    except (TypeError, IndexError):
        b_i = ones(N) * b_i
        h_i = ones(N) * h_i

    m = pe.ConcreteModel()
    m.i = pe.Set(initialize=range(N), ordered=True, doc='Training data set')
    m.j = pe.Set(initialize=range(p), ordered=True,  doc='Features set')
    m.q_j = pe.Var(m.j, within=pe.Reals, initialize=q_0, bounds=q_j_bounds, doc='Decision regressors')
    # m.q_j = pe.Var(m.j, within=pe.Reals, initialize={0: 0, 1: 1}, bounds=q_j_bounds, doc='Decision regressors')
    m.u_i = pe.Var(m.i, within=pe.NonNegativeReals, doc='Auxiliary var')
    m.o_i = pe.Var(m.i, within=pe.NonNegativeReals, doc='Auxiliary var')

    m.obj = pe.Objective(expr=1 / N * sum(b_i[i] * m.u_i[i] + h_i[i] * m.o_i[i] for i in m.i), sense=pe.minimize)
        # + lamb * sum(m.q_j[j]**2 for j in m.j), sense=pe.minimize)

    def con_rule_u_i(m, i):
        return m.u_i[i] >= y_i[i] - sum(m.q_j[j] * x_ij.iat[i, j] for j in m.j)

    def con_rule_o_i(m, i):
        return m.o_i[i] >= -1 * y_i[i] + sum(m.q_j[j] * x_ij.iat[i, j] for j in m.j)

    def con_rule_q(m, i):
        return y_bounds[0], sum(m.q_j[j] * x_ij.iat[i, j] for j in m.j), y_bounds[1]

    m.con1 = pe.Constraint(m.i, rule=con_rule_u_i, doc='Restriction u_i')
    m.con2 = pe.Constraint(m.i, rule=con_rule_o_i, doc='Restriction o_i')
    m.con3 = pe.Constraint(m.i, rule=con_rule_q, doc='Total power constrain')

    return m


def create_feasible_set_diameter_model(data):

    x_ij, y_i = data['x_ij'], data['y_i']
    y_bounds = data['y_bounds']
    q_j_bounds = data['q_j_bounds']
    N, p = shape(x_ij)

    m = pe.ConcreteModel()
    m.i = pe.Set(initialize=range(N), ordered=True, doc='Training data set')
    m.j = pe.Set(initialize=range(p), ordered=True,  doc='Features set')
    m.q1 = pe.Var(m.j, within=pe.Reals, bounds=q_j_bounds, doc='Decision regressors')
    m.q2 = pe.Var(m.j, within=pe.Reals, bounds=q_j_bounds, doc='Decision regressors')

    m.obj = pe.Objective(expr=sum(m.q1[j] ** 2 - 2 * m.q1[j] * m.q2[j] + m.q2[j] ** 2 for j in m.j), sense=pe.maximize)

    def con_rule_q1(m, i):
        return y_bounds[0], sum(m.q1[j] * x_ij.iat[i, j] for j in m.j), y_bounds[1]

    def con_rule_q2(m, i):
        return y_bounds[0], sum(m.q2[j] * x_ij.iat[i, j] for j in m.j), y_bounds[1]

    m.con1 = pe.Constraint(m.i, rule=con_rule_q1, doc='Total power constrain')
    m.con2 = pe.Constraint(m.i, rule=con_rule_q2, doc='Total power constrain')

    return m


def solve_optimization_model(optimization_model, config):
    """Auxiliary function to solve optimization_model with a certain config (dict).
    Pyomo Kernel Library is supported.
    """

    if config['verbose']:
        print('Solving optimization model...')

    # Saving or/and displaying the model before resolution.
    # Provides some information in case the resolution fails.
    pyomo_model_output(optimization_model, config)

    # Solving the model
    optimization_model, solver_output = solve_model(optimization_model, config)

    # Saving or/and displaying the model after resolution.
    # Overrides the previous model.
    pyomo_model_output(optimization_model, config)
    pyomo_solver_output(solver_output, config)

    # Check solver status.
    solver_status, solver_termination_condition, solver_summary = check_pyomo_model_status(
        solver_output, verbose=config['verbose'])

    # Gather solver status.
    solver_status = {
        "solver_summary": solver_summary, "solver_status": str(solver_status),
        "solver_termination_condition": str(solver_termination_condition)
    }

    # Extracts useful metrics from the resolution of the problem
    solver_additional_information = extract_additional_information(solver_output)

    return optimization_model, solver_status, solver_additional_information


def solve_model(model, config):
    """Solve a pyomo optimization model. Also support the NEOS optimization server."""

    solver_name = config['solver_name']
    solve_options = config['solve_options']
    solver_factory_options = config['solver_factory_options']

    if solver_name[0:4] == 'neos':
        with SolverManagerFactory('neos') as manager:
            solver = SolverFactory(solver_name[5:], **solver_factory_options)
            solver_output = manager.solve(model, opt=solver, **solve_options)
    else:
        solver = SolverFactory(solver_name, **solver_factory_options)
        solver_output = solver.solve(model, **solve_options)

    return model, solver_output


def check_pyomo_model_status(solver_output, verbose=False):
    """Check the solver status after solving the model."""

    solver_status = solver_output.solver.status
    solver_termination_condition = solver_output.solver.termination_condition

    if (solver_status == SolSt.ok) and (solver_termination_condition == TermCond.optimal):
        solver_summary = "optimal"
        if verbose:
            print("Optimal solution found.")
    elif solver_termination_condition == TermCond.infeasible:
        solver_summary = "infeasible"
        if verbose:
            print("Infeasible problem.")
    else:
        solver_summary = "other"
        if verbose:
            print(f"Error found different from infeasibility. Solver status: {solver_status}")

    return solver_status, solver_termination_condition, solver_summary


def pyomo_model_output(optimization_model, config):
    """Handle the display and writing of the pyomo model."""

    display_solver_info = config['display_info']
    write_solver_info = config['write_info']
    saving_path = config['saving_path']
    model_file_name = config['model_file_name']
    file_extension = config['file_extension']

    if display_solver_info:
        optimization_model.pprint()
    if write_solver_info:
        optimization_model.pprint(filename=join(saving_path, model_file_name + file_extension))


def pyomo_solver_output(solver_output, config):
    """Handle the display and writing of the solver output."""

    display_solver_info = config['display_info']
    write_solver_info = config['write_info']
    saving_path = config['saving_path']
    solver_file_name = config['solver_file_name']
    file_extension = config['file_extension']

    if display_solver_info:
        solver_output.write()

    if write_solver_info:
        with open(join(saving_path, solver_file_name + file_extension), 'w') as f:
            f.write('Solver output:\n\n')
            solver_output.write(ostream=f)


def extract_additional_information(solver_output):
    """Extract some relevant parameters from the solver output."""

    solver_additional_information = {}

    instructions = {
        'absolute_gap': 'solver_output.solution(0).gap',
        'lower_bound': 'solver_output.problem(0).lower_bound',
        'upper_bound': 'solver_output.problem(0).upper_bound',
        'time': 'solver_output.solver(0).user_time',
        'solver_status': 'solver_output.solver(0).status.value',
        'termination_condition': 'solver_output.solver(0).termination_condition.value',
        'relative_gap': 'solver_output.solution(0).gap / (1e-10 + abs(' +
            'solver_output.solution(0).objective[\'__default_objective__\'][\'Value\']))',
    }

    # Relative gap: Added 1e-10 as in CPLEX. Using default objective to be
    # valid minimizing or maximizing.

    for key, instruction in instructions.items():
        try:
            solver_additional_information[key] = eval(instruction)
        except (TypeError, IndexError):
            pass

    return solver_additional_information




