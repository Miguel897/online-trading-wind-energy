from os.path import join
import pyomo.environ as pe
from numpy import shape, ones
from pyomo.environ import SolverFactory, SolverManagerFactory
from pyomo.opt.results.solver import SolverStatus as SolSt, TerminationCondition as TermCond
from optimization_lp import solve_optimization_model, standard_solving_configuration

# Example of a solving configuration used in solve_optimization_model.
standard_solving_configuration = {
    'solver_name': 'cplex',
    'solve_options': {},
    'solver_factory_options': {'options': {
        'logfile': 'cplex.log',
        'output_clonelog': -1,
        'tee': True,
    }},
    'verbose': True,
    'display_info': True,
    'write_info': True,
    'solver_file_name': 'solver_info',
    'model_file_name': 'pyomo_model',
    'file_extension': '.txt',
    'saving_path': '',
    'timestamp': '',
}


def create_diameter_model(data):

    x_ij = data
    y_bounds = (0, 100)
    N, p = shape(x_ij)

    m = pe.ConcreteModel()
    m.i = pe.Set(initialize=range(N), ordered=True)
    m.j = pe.Set(initialize=range(p), ordered=True)
    m.q1_j = pe.Var(m.j, within=pe.Reals, bounds=(-1, 1))
    m.q2_j = pe.Var(m.j, within=pe.Reals, bounds=(-1, 1))

    m.obj = pe.Objective(expr=sum(m.q1_j[j]**2 - 2 * m.q1_j[j] * m.q2_j[j] + m.q2_j[j]**2 for j in m.j), sense=pe.maximize)

    def con_rule_q1(m, i):
        return y_bounds[0], sum(m.q1_j[j] * x_ij.iat[i, j] for j in m.j), y_bounds[1]

    def con_rule_q2(m, i):
        return y_bounds[0], sum(m.q2_j[j] * x_ij.iat[i, j] for j in m.j), y_bounds[1]

    m.con1 = pe.Constraint(m.i, rule=con_rule_q1, doc='Total power constrain')
    m.con2 = pe.Constraint(m.i, rule=con_rule_q2, doc='Total power constrain')

    return m


from functions_core import load_data
from config2_online_bidding import Label, Setting

_, _, _, x_data = load_data(Label, Setting)


x_data.reset_index(drop=True, inplace=True)
x_data = x_data.loc[0:10, :]
model = create_diameter_model(x_data)
model.pprint()
solved_model, solver_status, solver_additional_information = solve_optimization_model(model, standard_solving_configuration)
print(solver_status)