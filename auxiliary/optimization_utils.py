from os.path import join
from auxiliary.supress_log_file import custom_solver_factory
from pyomo.environ import SolverFactory, SolverManagerFactory
from pyomo.opt.results.solver import SolverStatus as SolSt, TerminationCondition as TermCond


def solve_optimization_model_direct(optimization_model, config, using_pyomo_kernel=False):
    """Auxiliary function to solve optimization_model with a certain config (dict).
    Pyomo Kernel Library is supported.
    """

    if config['verbose']:
        print('Solving optimization model...')

    # Saving or/and displaying the model before resolution.
    # Provides some information in case the resolution fails.
    pyomo_model_output(optimization_model, config, using_pyomo_kernel=using_pyomo_kernel)

    # Solving the model
    optimization_model, solver_output = solve_model(optimization_model, config)

    # Saving or/and displaying the model after resolution.
    # Overrides the previous model.
    pyomo_model_output(optimization_model, config, using_pyomo_kernel=using_pyomo_kernel)
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
    elif solver_name == 'cplex_direct':
        solver = custom_solver_factory('cplex_direct', **solver_factory_options)
        solver_output = solver.solve(model, **solve_options)
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


def pyomo_model_output(optimization_model, config, using_pyomo_kernel=False):
    """Handle the display and writing of the pyomo model."""

    display_solver_info = config['display_info']
    write_solver_info = config['write_info']
    saving_path = config['saving_path']
    model_file_name = config['model_file_name']
    file_extension = config['file_extension']

    if display_solver_info:
        if using_pyomo_kernel:
            readable_model = readable_pyomo_model(optimization_model)
            print(readable_model)
        else:
            optimization_model.pprint()
    if write_solver_info:
        with open(join(saving_path, model_file_name + file_extension), 'w') as f:
            if using_pyomo_kernel:
                f.write(readable_pyomo_model(optimization_model))
                optimization_model.write('model.lp', _solver_capability=None, _called_by_solver=False)
            else:
                # optimization_model.pprint(filename=join(saving_path, model_file_name + file_extension))
                optimization_model.pprint(ostream=f)


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
        'lower_bound': 'solver_output.problem(0).lower_bound',
        'upper_bound': 'solver_output.problem(0).upper_bound',
        'time': 'solver_output.solver(0).wallclock_time',
        'solver_status': 'solver_output.solver(0).status.value',
        'termination_condition': 'solver_output.solver(0).termination_condition.value',
        'absolute_gap': 'solver_output.problem(0).upper_bound - solver_output.problem(0).lower_bound',
        # 'relative_gap': '(solver_additional_information[\'upper_bound\']' +
        #                 '- solver_additional_information[\'lower_bound\'])' +
        #                 '/ (1e-10 + abs(solver_additional_information[\'upper_bound\']))',
        'relative_gap': '(solver_output.problem(0).upper_bound - solver_output.problem(0).lower_bound)' +
                        ' / (1e-10 + abs(solver_output.problem(0).upper_bound))',
    }

    for key, instruction in instructions.items():
        try:
            solver_additional_information[key] = eval(instruction)
        except (TypeError, IndexError):
            pass

    return solver_additional_information


def readable_pyomo_model(arg):
    return '.'