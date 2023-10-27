import cplex
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import os
import csv
import networkx as nx
from scipy.optimize import curve_fit
from switched_network_utils import *
import optimisation_switched
from sklearn import svm
from sklearn.model_selection import cross_val_score
from dill.source import getsource
import optimisation_no_switching
import Relaxed_heuristic
import LP_relaxation
# add_on_node_constraint - same
# add_capacity_requirement_constraint_multipath - same
# add_integer_limit - different - DONE
# add_cost_minimisation_objective - same
# add_maximal_single_path_capacity_multipath - different -DONE
# NEW CONSTRAINTS:
# add_path_detector_off_constraint:  \delta_{k}^{m} \leq \delta_{d}^{m} - DONE
# add loss of capacity due to calibration constraint:  q_{k}^{m} = q'_{k}^{m} - \frac{N}{c_{k}^{m}T}\delta_{k}^{m}
# CPLEX default Lower Bound is 0 => The following constraint is already applied.
# q_{k}^{m} \geq 0 - DONE



def split_sol_dict(sol_dict):
    """
    Split the solution dictionary into 4 dictionaries containing the fractional usage variables only and the binary
    variables only, lambda variables only and usage for keys variables
    Parameters
    ----------
    sol_dict : The solution dictionary containing solutions to the primary flow problem

    Returns : A dictionary with only the fractional detectors used, and a dictionary with only the binary values of
            whether the detector is on or off for cold and hot
    -------

    """
    use_dict = {}
    binary_dict = {}
    fraction_dict = {}
    lambda_dict = {}
    detector_useage = {}
    for key in sol_dict:
        # get all keys that are flow and add to dictionary
        if key[0] == "q":
            use_dict[key] = sol_dict[key]
        elif key[0] == "l":
            lambda_dict[key] = sol_dict[key]
        elif key[0] == "w":
            fraction_dict[key] = sol_dict[key]
        elif key[0] == "d":
            # get the keys that are binary 'on' 'off' and add to dictionary
            binary_dict[key] = sol_dict[key]
        else:
            detector_useage[key] = sol_dict[key]
    return use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage




def add_integer_limit(prob, hot_key_dict, cold_key_dict):
    """
    adds the constraint \sum_{i,j \in S} q'_{k=(i,j,d)}^{m} \leq \lambda_{d}^{m} - in the program we use w to mean q'
    """
    # get a dictionary with keys the detectors and a dictionary with (i,j,d)
    fraction_capacity_variables = {}
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    for k in hot_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for k in cold_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    # define the \lambda_{d}^{m} variables and add them to the list of variables
    capacity_integers_hot = []
    capacity_integers_cold = []
    detectors = []
    for key in fraction_capacity_variables:
        if key[2] not in detectors:
            capacity_integers_hot.append(f"lambda_{key[2]}^h")
            capacity_integers_cold.append(f"lambda_{key[2]}^c")
            detectors.append(key[2])
    prob.variables.add(names=capacity_integers_hot,
                       types=[prob.variables.type.integer] * len(capacity_integers_hot))
    prob.variables.add(names=capacity_integers_cold,
                       types=[prob.variables.type.integer] * len(capacity_integers_cold))


    capacity_variables_hot = []
    capacity_variables_cold = []
    for key in fraction_capacity_variables:
        capacity_variables_hot.append(f"w_{key[0], key[1], key[2]}^h")
        capacity_variables_cold.append(f"w_{key[0], key[1], key[2]}^c")
    prob.variables.add(names=capacity_variables_hot,
                       types=[prob.variables.type.continuous] * len(capacity_variables_hot),  lb = [0] * len(capacity_variables_hot))
    prob.variables.add(names=capacity_variables_cold,
                       types=[prob.variables.type.continuous] * len(capacity_variables_cold),  lb = [0] * len(capacity_variables_cold))
    # insert constraint
    lambda_cold = []
    lambda_hot = []
    q_cold = []
    q_hot = []
    new_detectors = []
    for key in fraction_capacity_variables:
        if key[2] not in new_detectors:
            # the constraint acts on the detector nodes - find all paths that pass through detector d and add these
            # to the network. Also add the lambdas into the array in the same element as the list of paths in the
            # q_cold, q_hot array are placed
            lambda_cold.append(f"lambda_{key[2]}^c")
            lambda_hot.append(f"lambda_{key[2]}^h")
            new_detectors.append(key[2])
            q_cold.append([f"w_{key[0],key[1],key[2]}^c"])
            q_hot.append([f"w_{key[0],key[1],key[2]}^h"])
        else:
            for i in range(len(new_detectors)):
                if key[2] == new_detectors[i]:
                    q_cold[i].append(f"w_{key[0],key[1],key[2]}^c")
                    q_hot[i].append(f"w_{key[0],key[1],key[2]}^h")
    for i in range(len(q_cold)):
        constraint = cplex.SparsePair(q_cold[i] + [lambda_cold[i]], val = [1] * len(q_cold[i]) + [-1])
        constraint_2 = cplex.SparsePair(q_hot[i] + [lambda_hot[i]], val = [1] * len(q_hot[i]) + [-1])
        prob.linear_constraints.add(lin_expr=[constraint, constraint_2], senses='LL', rhs = [0,0])



def add_maximal_single_path_capacity_multipath(prob, hot_key_dict, cold_key_dict, required_connections, KT):
    """
    adds the constraint to prevent any individual path from having a capacity greater than c_{min} - the constraint also
    ensures that if the path is off then q' is 0 and not positive, KT is the fraction N/T that is the ratio of time
    required to calibrate the equiptment after switching.
    """
    fraction_capacity_variables = {}
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    for k in hot_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for k in cold_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    # add the dinaries \delta_{k}^{m} to the problem
    path_on_hot = []
    path_on_cold = []
    for key in fraction_capacity_variables:
        path_on_hot.append(f"Delta_{key[0],key[1],key[2]}^h")
        path_on_cold.append(f"Delta_{key[0],key[1],key[2]}^c")
    prob.variables.add(names=path_on_hot,
                       types=[prob.variables.type.binary] * len(path_on_hot))
    prob.variables.add(names=path_on_cold,
                       types=[prob.variables.type.binary] * len(path_on_cold))
    for connection in required_connections:
        for detector in detectors:
            ind_flow_hot = [f"w_{connection[0], connection[1], detector}^h", f"Delta_{connection[0],connection[1],detector}^h"]
            ind_flow_cold = [f"w_{connection[0], connection[1], detector}^c", f"Delta_{connection[0],connection[1],detector}^c"]
            ### required_connections[connection] * cmin ??????? - here we have required_connections[connection] = cmin
            # and M * cmin is the value in add_capacity_requirement_constraint_multipath
            capacity_hot = [float(hot_key_dict[(connection[0], connection[1], detector)]), float(-required_connections[connection] - KT)]
            capacity_cold = [float(cold_key_dict[(connection[0], connection[1], detector)]), float(-required_connections[connection] - KT)]
            if capacity_hot[0] > 0.00001 and capacity_cold[0]> 0.000001:
                constraints = [cplex.SparsePair(ind = ind_flow_hot, val = capacity_hot), cplex.SparsePair(ind = ind_flow_cold, val = capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='LL', rhs=[0,0])
            elif capacity_cold[0] > 0.000001:
                constraints = [cplex.SparsePair(ind = ind_flow_cold, val = capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='L',
                                            rhs=[0])


def add_path_detector_off_constraint(prob, hot_key_dict, cold_key_dict):
    fraction_capacity_variables = {}
    for k in hot_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for k in cold_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for key in fraction_capacity_variables:
        variables_hot =  [f"Delta_{key[0], key[1], key[2]}^h", f"delta_{key[2]}^h"]
        variables_cold = [f"Delta_{key[0], key[1], key[2]}^c", f"delta_{key[2]}^c"]
        constraints = [cplex.SparsePair(ind=variables_hot, val=[1,-1]),
                       cplex.SparsePair(ind=variables_cold, val =[1,-1])]
        prob.linear_constraints.add(lin_expr=constraints, senses='LL', rhs=[0, 0])


def add_loss_of_capacity_due_to_calibration_constraint(prob, hot_key_dict, cold_key_dict, KT):
    fraction_capacity_variables = {}
    for k in hot_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for k in cold_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for key in fraction_capacity_variables:
        variables_hot = [f"q_{key[0], key[1], key[2]}^h", f"w_{key[0],key[1],key[2]}^h",f"Delta_{key[0],key[1],key[2]}^h"]
        variables_cold = [f"q_{key[0], key[1], key[2]}^c", f"w_{key[0], key[1], key[2]}^c",
                         f"Delta_{key[0], key[1], key[2]}^c"]
        capacity_hot = hot_key_dict[(key[0], key[1], key[2])]
        capacity_cold = cold_key_dict[(key[0], key[1], key[2])]
        if capacity_hot > 0.000001:
            constraints = [cplex.SparsePair(ind=variables_hot, val=[1, -1, KT/capacity_hot])]
            prob.linear_constraints.add(lin_expr=constraints, senses='E', rhs=[0])
        else:
            # if capacity of path = 0 then constraint becomes pair of constraints (delta_{k}^m = 0, q_{k}^{m}= q'_{k}^{m})
            delta_hot = [f"Delta_{key[0],key[1],key[2]}^h"]
            other_terms = [f"q_{key[0], key[1], key[2]}^h", f"w_{key[0],key[1],key[2]}^h"]
            constraints = [cplex.SparsePair(ind=delta_hot, val=[1]), cplex.SparsePair(ind = other_terms, val = [1,-1])]
            prob.linear_constraints.add(lin_expr = constraints, senses = "EE", rhs = [0,0])
        if capacity_cold > 0.000001:
            constraints = [cplex.SparsePair(ind=variables_cold, val=[1, -1, KT / capacity_cold])]
            prob.linear_constraints.add(lin_expr=constraints, senses='E', rhs=[0])
        else:
            # if capacity of path = 0 then constraint becomes pair of constraints (delta_{k}^m = 0, q_{k}^{m}= q'_{k}^{m})
            delta_cold = [f"Delta_{key[0], key[1], key[2]}^c"]
            other_terms = [f"q_{key[0], key[1], key[2]}^c", f"w_{key[0], key[1], key[2]}^c"]
            constraints = [cplex.SparsePair(ind=delta_cold, val=[1]), cplex.SparsePair(ind=other_terms, val=[1, -1])]
            prob.linear_constraints.add(lin_expr=constraints, senses="EE", rhs=[0,0])


def add_integer_limit_no_w_terms(prob, hot_key_dict, cold_key_dict, fract_switch):
    """
    adds the constraint \sum_{i,j \in S} Q_{k=(i,j,d)}^{m} \leq (1-fract_switch) *\lambda_{d}^{m} - in the program we use w to mean q'
    """
    # get a dictionary with keys the detectors and a dictionary with (i,j,d)
    fraction_capacity_variables = {}
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    for k in hot_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for k in cold_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    # define the \lambda_{d}^{m} variables and add them to the list of variables
    capacity_integers_hot = []
    capacity_integers_cold = []
    detectors = []
    for key in fraction_capacity_variables:
        if key[2] not in detectors:
            capacity_integers_hot.append(f"lambda_{key[2]}^h")
            capacity_integers_cold.append(f"lambda_{key[2]}^c")
            detectors.append(key[2])
    prob.variables.add(names=capacity_integers_hot,
                       types=[prob.variables.type.integer] * len(capacity_integers_hot))
    prob.variables.add(names=capacity_integers_cold,
                       types=[prob.variables.type.integer] * len(capacity_integers_cold))

    # insert constraint
    lambda_cold = []
    lambda_hot = []
    q_cold = []
    q_hot = []
    new_detectors = []
    for key in fraction_capacity_variables:
        if key[2] not in new_detectors:
            # the constraint acts on the detector nodes - find all paths that pass through detector d and add these
            # to the network. Also add the lambdas into the array in the same element as the list of paths in the
            # q_cold, q_hot array are placed
            lambda_cold.append(f"lambda_{key[2]}^c")
            lambda_hot.append(f"lambda_{key[2]}^h")
            new_detectors.append(key[2])
            q_cold.append([f"q_{key[0],key[1],key[2]}^c"])
            q_hot.append([f"q_{key[0],key[1],key[2]}^h"])
        else:
            for i in range(len(new_detectors)):
                if key[2] == new_detectors[i]:
                    q_cold[i].append(f"q_{key[0],key[1],key[2]}^c")
                    q_hot[i].append(f"q_{key[0],key[1],key[2]}^h")
    for i in range(len(q_cold)):
        constraint = cplex.SparsePair(q_cold[i] + [lambda_cold[i]], val = [1] * len(q_cold[i]) + [-(1-fract_switch)])
        constraint_2 = cplex.SparsePair(q_hot[i] + [lambda_hot[i]], val = [1] * len(q_hot[i]) + [-(1-fract_switch)])
        prob.linear_constraints.add(lin_expr=[constraint, constraint_2], senses='LL', rhs = [0,0])


def add_maximal_single_path_capacity_multipath_no_w_term(prob, hot_key_dict, cold_key_dict, required_connections):
    """
    adds the constraint to prevent any individual path from having a capacity greater than c_{min} - the constraint also
    ensures that if the path is off then q' is 0 and not positive, KT is the fraction N/T that is the ratio of time
    required to calibrate the equiptment after switching.
    """
    fraction_capacity_variables = {}
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    for k in hot_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for k in cold_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    # add the dinaries \delta_{k}^{m} to the problem
    path_on_hot = []
    path_on_cold = []
    for key in fraction_capacity_variables:
        path_on_hot.append(f"Delta_{key[0],key[1],key[2]}^h")
        path_on_cold.append(f"Delta_{key[0],key[1],key[2]}^c")
    prob.variables.add(names=path_on_hot,
                       types=[prob.variables.type.binary] * len(path_on_hot))
    prob.variables.add(names=path_on_cold,
                       types=[prob.variables.type.binary] * len(path_on_cold))
    for connection in required_connections:
        for detector in detectors:
            ind_flow_hot = [f"q_{connection[0], connection[1], detector}^h", f"Delta_{connection[0],connection[1],detector}^h"]
            ind_flow_cold = [f"q_{connection[0], connection[1], detector}^c", f"Delta_{connection[0],connection[1],detector}^c"]
            ### required_connections[connection] * cmin ??????? - here we have required_connections[connection] = cmin
            # and M * cmin is the value in add_capacity_requirement_constraint_multipath
            capacity_hot = [float(hot_key_dict[(connection[0], connection[1], detector)]), float(-required_connections[connection])]
            capacity_cold = [float(cold_key_dict[(connection[0], connection[1], detector)]), float(-required_connections[connection])]
            if capacity_hot[0] > 0.00001 and capacity_cold[0]> 0.000001:
                constraints = [cplex.SparsePair(ind = ind_flow_hot, val = capacity_hot), cplex.SparsePair(ind = ind_flow_cold, val = capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='LL', rhs=[0,0])
            elif capacity_cold[0] > 0.000001:
                constraints = [cplex.SparsePair(ind = ind_flow_cold, val = capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='L',
                                            rhs=[0])


def add_maximal_single_path_capacity_multipath_no_w_term_no_delta_term(prob, hot_key_dict, cold_key_dict, required_connections):
    """
    adds the constraint to prevent any individual path from having a capacity greater than c_{min} - the constraint also
    ensures that if the path is off then q' is 0 and not positive, KT is the fraction N/T that is the ratio of time
    required to calibrate the equiptment after switching.
    """
    fraction_capacity_variables = {}
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    for k in hot_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for k in cold_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
        elif k not in fraction_capacity_variables:
            fraction_capacity_variables[k] = 1
    for connection in required_connections:
        for detector in detectors:
            ind_flow_hot = [f"q_{connection[0], connection[1], detector}^h"]
            ind_flow_cold = [f"q_{connection[0], connection[1], detector}^c"]
            ### required_connections[connection] * cmin ??????? - here we have required_connections[connection] = cmin
            # and M * cmin is the value in add_capacity_requirement_constraint_multipath
            capacity_hot = [float(hot_key_dict[(connection[0], connection[1], detector)])]
            capacity_cold = [float(cold_key_dict[(connection[0], connection[1], detector)])]
            if capacity_hot[0] > 0.00001 and capacity_cold[0]> 0.000001:
                constraints = [cplex.SparsePair(ind = ind_flow_hot, val = capacity_hot), cplex.SparsePair(ind = ind_flow_cold, val = capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='LL', rhs=[float(required_connections[connection]), float(required_connections[connection])])
            elif capacity_cold[0] > 0.000001:
                constraints = [cplex.SparsePair(ind = ind_flow_cold, val = capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='L',
                                            rhs=[float(required_connections[connection])])



def initial_optimisation_switching(hot_key_dict, cold_key_dict, required_connections, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, KT, time_limit = 1e7, early_stop = None):
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    optimisation_switched.add_capacity_requirement_constraint_multipath(prob, hot_key_dict, cold_key_dict, required_connections, M)
    add_integer_limit(prob, hot_key_dict, cold_key_dict)
    optimisation_switched.add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N)
    add_maximal_single_path_capacity_multipath(prob, hot_key_dict, cold_key_dict, required_connections, KT)
    add_path_detector_off_constraint(prob, hot_key_dict, cold_key_dict)
    add_loss_of_capacity_due_to_calibration_constraint(prob, hot_key_dict, cold_key_dict, KT)
    optimisation_switched.add_cost_minimisation_objective(prob, hot_key_dict, cold_key_dict, cost_det_h, cost_det_c,
                                    cost_on_h, cost_on_c)
    prob.write("test_switched.lp")
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.dual)
    if early_stop is not None:
        prob.parameters.mip.tolerances.mipgap.set(early_stop)
    # prob.parameters.mip.limits.cutpasses.set(1)
    # prob.parameters.mip.strategy.probe.set(-1)
    # prob.parameters.mip.strategy.variableselect.set(4)
    print(prob.parameters.get_changed())
    prob.parameters.timelimit.set(time_limit)
    t_1 = time.time()
    print("Time to set up problem: " + str(t_1 - t_0))
    prob.solve()
    t_2 = time.time()
    print("Time to solve problem: " + str(t_2 - t_1))
    print(f"The Minimum Number of Trusted Nodes: {prob.solution.get_objective_value()}")
    print(f"Number of Variables = {prob.variables.get_num()}")
    print(f"Number of Conditions = {prob.linear_constraints.get_num()}")
    sol_dict = optimisation_switched.create_sol_dict(prob)

    return sol_dict, prob


def initial_optimisation_switching_fixed_switching_ratio(hot_key_dict, cold_key_dict, required_connections, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, fract_switch = 0.9, time_limit = 1e7, early_stop = None):
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    optimisation_switched.add_capacity_requirement_constraint_multipath(prob, hot_key_dict, cold_key_dict, required_connections, M)
    add_integer_limit_no_w_terms(prob, hot_key_dict, cold_key_dict, fract_switch)
    optimisation_switched.add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N)
    add_maximal_single_path_capacity_multipath_no_w_term(prob, hot_key_dict, cold_key_dict, required_connections)
    add_path_detector_off_constraint(prob, hot_key_dict, cold_key_dict)
    optimisation_switched.add_cost_minimisation_objective(prob, hot_key_dict, cold_key_dict, cost_det_h, cost_det_c,
                                    cost_on_h, cost_on_c)
    prob.write("test_switched.lp")
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.dual)
    if early_stop is not None:
        prob.parameters.mip.tolerances.mipgap.set(early_stop)
    # prob.parameters.mip.limits.cutpasses.set(1)
    # prob.parameters.mip.strategy.probe.set(-1)
    # prob.parameters.mip.strategy.variableselect.set(4)
    print(prob.parameters.get_changed())
    prob.parameters.timelimit.set(time_limit)
    t_1 = time.time()
    print("Time to set up problem: " + str(t_1 - t_0))
    prob.solve()
    t_2 = time.time()
    print("Time to solve problem: " + str(t_2 - t_1))
    print(f"The Minimum Number of Trusted Nodes: {prob.solution.get_objective_value()}")
    print(f"Number of Variables = {prob.variables.get_num()}")
    print(f"Number of Conditions = {prob.linear_constraints.get_num()}")
    sol_dict = optimisation_switched.create_sol_dict(prob)

    return sol_dict, prob, t_2-t_1



def initial_optimisation_switching_fixed_switching_ratio_no_delta(hot_key_dict, cold_key_dict, required_connections, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, fract_switch = 0.9, time_limit = 1e7, early_stop = None):
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    optimisation_switched.add_capacity_requirement_constraint_multipath(prob, hot_key_dict, cold_key_dict, required_connections, M)
    add_integer_limit_no_w_terms(prob, hot_key_dict, cold_key_dict, fract_switch)
    optimisation_switched.add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N)
    add_maximal_single_path_capacity_multipath_no_w_term_no_delta_term(prob, hot_key_dict, cold_key_dict, required_connections)
    optimisation_switched.add_cost_minimisation_objective(prob, hot_key_dict, cold_key_dict, cost_det_h, cost_det_c,
                                    cost_on_h, cost_on_c)
    prob.write("test_switched.lp")
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.dual)
    if early_stop is not None:
        prob.parameters.mip.tolerances.mipgap.set(early_stop)
    # prob.parameters.mip.limits.cutpasses.set(1)
    # prob.parameters.mip.strategy.probe.set(-1)
    # prob.parameters.mip.strategy.variableselect.set(4)
    print(prob.parameters.get_changed())
    prob.parameters.timelimit.set(time_limit)
    t_1 = time.time()
    print("Time to set up problem: " + str(t_1 - t_0))
    prob.solve()
    t_2 = time.time()
    print("Time to solve problem: " + str(t_2 - t_1))
    print(f"The Minimum Number of Trusted Nodes: {prob.solution.get_objective_value()}")
    print(f"Number of Variables = {prob.variables.get_num()}")
    print(f"Number of Conditions = {prob.linear_constraints.get_num()}")
    sol_dict = optimisation_switched.create_sol_dict(prob)

    return sol_dict, prob, t_2-t_1

def find_critical_ratio_cooled_uncooled(hot_bob_capacity_file, cold_bob_capacity_file, graph_edge_data_file, graph_node_data_file, N, M,f_switch, cost_det_h, cost_on_h, cmin=5000, data_storage_location_keep_each_loop = None, complete = False):
    # things that affect critical ratio: number of nodes and connectivity of graph
    if not complete:
        hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
            hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
        # graphs = import_graph_structure(node_information = graph_node_data_file, edge_information = graph_edge_data_file)
        cold_nodes_dict = {}
        hot_nodes_dict = {}
        no_soln_set = []
        ### if input storange file =! None then figure out where to start from
        if data_storage_location_keep_each_loop != None:
            if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
                last_row_explored = plot_information.iloc[[-1]]
                last_ratio_done = last_row_explored["ratio_hot_cold"]
                dataframe_of_last_ratio_done = plot_information[plot_information["ratio_hot_cold"] == last_ratio_done.iloc[0]]
                number_of_rows = len(dataframe_of_last_ratio_done.index)
                last_ratio_done = last_ratio_done.iloc[0]
            else:
                last_ratio_done = 0.5
                number_of_rows = 0
                dictionary_fieldnames = ["ratio_hot_cold","number_nodes","cold_nodes_on","hot_nodes_on"]
                with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                    writer.writeheader()
        else:
            last_ratio_done = 0.5
            number_of_rows = 0
        for ratio_hot_cold in np.arange(last_ratio_done,5,0.2):
            cost_det_c = cost_det_h * ratio_hot_cold
            cost_on_c = cost_on_h * ratio_hot_cold
            cold_node_on_number_nodes = {}
            hot_node_on_number_nodes = {}
            for key in hot_capacity_dict.keys():
                if number_of_rows != 0:
                    sol_dict, prob, time_1 = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                                                        hot_key_dict=hot_capacity_dict[key],
                                                        cold_key_dict=cold_capacity_dict[key],
                                                        required_connections=required_connections[key],
                                                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                        fract_switch=f_switch,
                                                        time_limit=5e3, early_stop=0.003)
                    if optimisation_switched.check_solvable(prob):
                        number_of_rows -= 1
                    continue
                if key not in no_soln_set:
                    try:
                        # t_0 = time.time()
                        sol_dict, prob, time_1 = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                                                        hot_key_dict=hot_capacity_dict[key],
                                                        cold_key_dict=cold_capacity_dict[key],
                                                        required_connections=required_connections[key],
                                                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                        fract_switch=f_switch,
                                                        time_limit=5e3, early_stop=0.003)
                        if optimisation_switched.check_solvable(prob):
                            q_solns, lambda_soln, w_solns, binary_solns, detector_usage_soln = split_sol_dict(sol_dict)
                            number_nodes = optimisation_switched.get_number_of_nodes(hot_capacity_dict[key], cold_capacity_dict[key])
                            cold_binaries = list(optimisation_switched.get_cold_binaries(binary_solns).values())
                            hot_binaries = list(optimisation_switched.get_hot_binaries(binary_solns).values())
                            cold_nodes_on = sum(cold_binaries) / (sum(cold_binaries) + sum(hot_binaries))
                            hot_nodes_on = sum(hot_binaries) / (sum(cold_binaries) + sum(hot_binaries))
                            if number_nodes not in cold_node_on_number_nodes.keys():
                                cold_node_on_number_nodes[number_nodes] = [cold_nodes_on]
                                hot_node_on_number_nodes[number_nodes]  = [hot_nodes_on]
                            else:
                                cold_node_on_number_nodes[number_nodes].append(cold_nodes_on)
                                hot_node_on_number_nodes[number_nodes].append(hot_nodes_on)
                            if data_storage_location_keep_each_loop != None:
                                dictionary = [{"ratio_hot_cold": ratio_hot_cold, "number_nodes": number_nodes , "cold_nodes_on": cold_nodes_on, "hot_nodes_on": hot_nodes_on}]
                                dictionary_fieldnames = ["ratio_hot_cold", "number_nodes", "cold_nodes_on", "hot_nodes_on"]
                                if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                        writer.writerows(dictionary)
                                else:
                                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                        writer.writeheader()
                                        writer.writerows(dictionary)
                    except:
                        no_soln_set.append(key)
                        print("Error in the optimisation")
            if data_storage_location_keep_each_loop == None:
                for number_nodes in cold_node_on_number_nodes.keys():
                    cold_nodes_dict[ratio_hot_cold, number_nodes] = cold_node_on_number_nodes[number_nodes]
                    hot_nodes_dict[ratio_hot_cold, number_nodes] = hot_node_on_number_nodes[number_nodes]
    # need to separate range into 3 regions - cold only, both cold and hot and hot only
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        cold_nodes_dict = {}
        hot_nodes_dict = {}
        for index, row in plot_information.iterrows():
            if (row["ratio_hot_cold"], row["number_nodes"]) not in hot_nodes_dict.keys():
                hot_nodes_dict[(row["ratio_hot_cold"], row["number_nodes"])] = [row["hot_nodes_on"]]
                cold_nodes_dict[(row["ratio_hot_cold"], row["number_nodes"])] = [row["cold_nodes_on"]]
            else:
                hot_nodes_dict[(row["ratio_hot_cold"], row["number_nodes"])].append(row["hot_nodes_on"])
                cold_nodes_dict[(row["ratio_hot_cold"], row["number_nodes"])].append(row["cold_nodes_on"])


    for key in cold_nodes_dict.keys():
        cold_nodes_dict[key] = sum(cold_nodes_dict[key])/len(cold_nodes_dict[key])
        hot_nodes_dict[key] = sum(hot_nodes_dict[key])/len(hot_nodes_dict[key])
    cmap = {}
    for key in cold_nodes_dict.keys():
        cold_values = cold_nodes_dict[key]
        hot_values = hot_nodes_dict[key]
        cmap[key] = cold_values - hot_values
        # cmap[key] = np.array([hot_values, 0, cold_values, 1])
    x = []
    y = []
    cmap_array = []
    for key in cmap.keys():
        x.append(key[0])
        y.append(key[1])
        cmap_array.append(cmap[key])
    plt.style.use('dark_background')
    norm = mcolors.Normalize(vmin=-1,vmax=1)
    plt.scatter(x, y, c=cmap_array, norm =norm, cmap = "RdBu")
    cbar = plt.colorbar()
    cbar.set_label(r"$\dfrac{(\sum_{i \in C} \delta_{i}^{c}- \delta_{i}^{h})}{(\sum_{i \in C} \delta_{i}^{c}+ \delta_{i}^{h})}$")
    plt.xlabel("Ratio of Cost of Cooled Detectors to Uncooled Detectors")
    plt.ylabel("Number of Nodes in the Graph")
    plt.savefig("heat_map_mesh_topology_35_nodes_on_7.png")
    plt.show()

def KT_required_to_make_significant_difference(hot_bob_capacity_file, cold_bob_capacity_file, graph_edge_data_file, graph_node_data_file, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, cmin=5000, data_storage_location_keep_each_loop = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    no_soln_set = []
    differences = {}
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["KT"]
            dataframe_of_last_ratio_done = plot_information[plot_information["KT"] == last_ratio_done.iloc[0]]
            number_of_rows = len(dataframe_of_last_ratio_done.index)
            KT_current = last_ratio_done.iloc[0]
        else:
            KT_current = 0
            number_of_rows = 0
            dictionary_fieldnames = ["KT","number_nodes","differences_switching_no_switching"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        KT_current = 0
        number_of_rows = 0
    for KT in np.arange(KT_current, 5000, 500):
        differences_switch_no_switch = {}
        for key in hot_capacity_dict.keys():
            if number_of_rows != 0:
                sol_dict, prob = initial_optimisation_switching(hot_key_dict=hot_capacity_dict[key],
                                                                cold_key_dict=cold_capacity_dict[key],
                                                                required_connections=required_connections[key],
                                                                cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                KT=KT,
                                                                time_limit=5e2, early_stop=0.003)
                if optimisation_switched.check_solvable(prob):
                    number_of_rows -= 1
                    continue
                else:
                    no_soln_set.append(key)
            if key not in no_soln_set:
                try:
                    t_0 = time.time()
                    sol_dict, prob = initial_optimisation_switching(hot_key_dict=hot_capacity_dict[key],
                                                                    cold_key_dict=cold_capacity_dict[key],
                                                                    required_connections=required_connections[key],
                                                                    cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                    cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                    KT=KT,
                                                                    time_limit=5e2, early_stop=0.003)

                    sol_dict_no_switching, prob_no_switching = optimisation_switched.initial_optimisation_multiple_paths(hot_key_dict=hot_capacity_dict[key],
                                                                         cold_key_dict=cold_capacity_dict[key],
                                                                         required_connections=required_connections[key],
                                                                         cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                         cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N,
                                                                         M=M,
                                                                         time_limit=5e2, early_stop=0.003)

                    if optimisation_switched.check_solvable(prob) and optimisation_switched.check_solvable(prob_no_switching):
                        # +ve if switching solution costs more than ignoring switching -
                        # -ve if switching solution costs less than ignoring switching
                        difference_no_switching = (prob.solution.get_objective_value() - prob_no_switching.solution.get_objective_value()) / prob.solution.get_objective_value()
                        number_nodes = optimisation_switched.get_number_of_nodes(hot_capacity_dict[key], cold_capacity_dict[key])
                        if number_nodes not in differences_switch_no_switch.keys():
                            differences_switch_no_switch[number_nodes] = [difference_no_switching]
                        else:
                            differences_switch_no_switch[number_nodes].append(difference_no_switching)
                        if data_storage_location_keep_each_loop != None:
                            dictionary = [{"KT": KT, "number_nodes": number_nodes , "differences_switching_no_switching": difference_no_switching}]
                            dictionary_fieldnames = ["KT", "number_nodes", "differences_switching_no_switching"]
                            if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                                with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writerows(dictionary)
                            else:
                                with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writeheader()
                                    writer.writerows(dictionary)

                except:
                    no_soln_set.append(key)
                    print("Error in the optimisation")
        if data_storage_location_keep_each_loop == None:
            for number_nodes in differences_switch_no_switch.keys():
                differences[KT, number_nodes] = differences_switch_no_switch[number_nodes]
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        differences = {}
        for index, row in plot_information.iterrows():
            if (row["KT"], row["number_nodes"]) not in differences.keys():
                differences[(row["KT"], row["number_nodes"])] = [row["differences_switching_no_switching"]]
            else:
                differences[(row["KT"], row["number_nodes"])] .append(row["differences_switching_no_switching"])

    mean_differences = {}
    std_differences = {}
    for key in differences.keys():
        mean_differences[key] = np.mean(differences[key])
        std_differences[key] = np.std(differences[key])
    mean_differences_no_nodes = {}
    std_differences_no_nodes = {}
    x = {}
    for key in mean_differences.keys():
        if key[1] not in mean_differences_no_nodes.keys():
            mean_differences_no_nodes[key[1]] = [mean_differences[key]]
            std_differences_no_nodes[key[1]] = [std_differences[key]]
            x[key[1]] = [key[0]]
        else:
            mean_differences_no_nodes[key[1]].append(mean_differences[key])
            std_differences_no_nodes[key[1]].append(std_differences[key])
            x[key[1]].append(key[0])
    for key in x.keys():
        plt.errorbar(x[key], mean_differences_no_nodes[key], yerr = std_differences_no_nodes[key])
        plt.xlabel("N/T (1/s) for calibration")
        plt.ylabel("(Calibration - Ignoring Calibration)/(Calibration)")
        plt.savefig("calibration_graph_mesh_topology_35_node_number" + str(key) + ".png")
        plt.show()





def difference_in_solution_costs_for_various_topologies(hot_bob_capacity_file_list, cold_bob_capacity_file_list, graph_edge_data_file_list, graph_node_data_file_list, topologies_list, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, KT, cmin=5000, data_storage_location_keep_each_loop = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances, graphs = {}, {}, {}, {}, {}
    for i in range(len(hot_bob_capacity_file_list)):
        hot_capacity_dict[topologies_list[i]], cold_capacity_dict[topologies_list[i]], required_connections[topologies_list[i]], distances[topologies_list[i]] = import_switched_network_values_multiple_graphs(hot_bob_capacity_file=hot_bob_capacity_file_list[i],
                                                       cold_bob_capacity_file=cold_bob_capacity_file_list[i], cmin=cmin)
        graphs[topologies_list[i]] = import_graph_structure(node_information=graph_node_data_file_list[i], edge_information=graph_edge_data_file_list[i])


    no_soln_set = []
    objective_values = {}
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["topology"]
            dataframe_of_topology_done = plot_information[ plot_information["topology"] == last_ratio_done.iloc[0]]
            number_of_rows = len(dataframe_of_topology_done.index)
            topology_current = last_ratio_done.iloc[0]
        else:
            topology_current = None
            number_of_rows = 0
            dictionary_fieldnames = ["ratio_hot_cold", "number_nodes", "cold_nodes_on", "hot_nodes_on"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        topology_current = None
        number_of_rows = 0
    for key in hot_capacity_dict.keys():
        # skip till the topology we have done - after that set topology_current to None to stop checking
        if topology_current != None:
            if key != topology_current:
                continue
            else:
                topology_current = None
        objective_values_topology = {}
        for key_graph in hot_capacity_dict[key].keys():
            if number_of_rows != 0:
                number_of_rows -= 1
                continue
            if key not in no_soln_set:
                try:
                    t_0 = time.time()
                    sol_dict, prob = initial_optimisation_switching(hot_key_dict=hot_capacity_dict[key][key_graph],
                                                                    cold_key_dict=cold_capacity_dict[key][key_graph],
                                                                    required_connections=required_connections[key][key_graph],
                                                                    cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                    cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                    KT=KT,
                                                                    time_limit=5e2, early_stop=0.003)
                    if optimisation_switched.check_solvable(prob) :
                        objective_value = prob.solution.get_objective_value()
                        number_nodes = optimisation_switched.get_number_of_nodes(hot_capacity_dict[key], cold_capacity_dict[key])
                        if number_nodes not in objective_values_topology.keys():
                            objective_values_topology[number_nodes] = [objective_value]
                        else:
                            objective_values_topology[number_nodes].append(objective_value)
                        if data_storage_location_keep_each_loop != None:
                            dictionary = [{"topology": key, "number_nodes": number_nodes , "objective_value": objective_value}]
                            dictionary_fieldnames = ["topology", "number_nodes", "objective_value"]
                            if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                                with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writerows(dictionary)
                            else:
                                with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writeheader()
                                    writer.writerows(dictionary)

                except:
                    no_soln_set.append(key)
                    print("Error in the optimisation")
        if data_storage_location_keep_each_loop == None:
            for number_nodes in objective_values_topology.keys():
                objective_values[key, number_nodes] = objective_values_topology[number_nodes]
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        objective_values = {}
        for index, row in plot_information.iterrows():
            if (row["topology"], row["number_nodes"]) not in objective_values.keys():
                objective_values[(row["topology"], row["number_nodes"])] = [row["objective_value"]]
            else:
                objective_values[(row["topology"], row["number_nodes"])].append(row["objective_value"])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences_no_nodes = {}
    std_differences_no_nodes = {}
    # topologies
    x = {}
    for key in mean_objectives.keys():
        if key[1] not in mean_differences_no_nodes.keys():
            mean_differences_no_nodes[key[1]] = [mean_objectives[key]]
            std_differences_no_nodes[key[1]] = [std_objectives[key]]
            x[key[1]] = [key[0]]
        else:
            mean_differences_no_nodes[key[1]].append(mean_objectives[key])
            std_differences_no_nodes[key[1]].append(std_objectives[key])
            x[key[1]].append(key[0])


def time_taken_with_increasing_number_of_nodes(hot_bob_capacity_file, cold_bob_capacity_file,  graph_edge_data_file, graph_node_data_file, cost_det_h, cost_det_c, cost_on_h,
                                      cost_on_c, N, M, frac_switch, cmin=5000):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)

    time_taken = {}
    for key in hot_capacity_dict.keys():
        try:
            sol_dict, prob, time_to_solve = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                hot_key_dict=hot_capacity_dict[key],
                cold_key_dict=cold_capacity_dict[key],
                required_connections=required_connections[key],
                cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                fract_switch=frac_switch,
                time_limit=5e3, early_stop=0.003)
            if optimisation_switched.check_solvable(prob):
                use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = split_sol_dict(sol_dict)
                number_nodes = optimisation_switched.get_number_of_nodes(hot_capacity_dict[key], cold_capacity_dict[key])
                optimisation_switched.plot_graph(graphs[key], binary_dict)
                if number_nodes in time_taken.keys():
                    time_taken[number_nodes].append(time_to_solve)
                else:
                    time_taken[number_nodes] = [time_to_solve]
        except:
            continue
    time_costs_mean_std = {}
    x = []
    y = []
    yerr = []
    for key in time_taken:
        time_costs_mean_std[key] = [np.mean(time_taken[key]), np.std(time_taken[key])]
        x.append(key)
        y.append(time_costs_mean_std[key][0])
        yerr.append(time_costs_mean_std[key][1])
        # x                     y               yerr
    # fit of exponential curve to initial points
    # x_exponential = x
    # y_exponential = y
    # popt, pcov = curve_fit(optimisation_switched.exponential_fit, x_exponential, y_exponential)
    # x_exponential = np.arange(x[0], x[-1], 0.1)
    # y_exponential = [optimisation_switched.exponential_fit(a, popt[0], popt[1], popt[2]) for a in x_exponential]
    # # fit of polynomial
    # popt_poly, pcov_poly = curve_fit(optimisation_switched.polynomial, x, y)
    # y_poly = [optimisation_switched.polynomial(a, popt_poly[0], popt_poly[1]) for a in x_exponential]

    plt.errorbar(x, y, yerr=yerr, color="r")
    # plt.plot(x_exponential[:int(np.ceil(len(x_exponential)/1.25))], y_exponential[:int(np.ceil(len(x_exponential)/1.25))], color = "b")
    # plt.plot(x_exponential, y_poly, color = "k")
    plt.xlabel("No. Detector Sites in Graph", fontsize=10)
    plt.ylabel("Time/s", fontsize=10)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("time_investigation_mesh_topology_detector_sites.png")
    plt.show()


def switch_loss_cost_comparison(cost_det_h, cost_det_c, cost_on_h,
                                      cost_on_c, N, M, frac_switch, cmin=5000, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_no_switch = None):
    objective_values_switch_loss = {}
    objective_value_at_1_dB = {}
    no_solution_list = []
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["db_switch"]
            dataframe_of_fswitch_done = plot_information[plot_information["db_switch"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            dbswitch_current = last_ratio_done.iloc[0]
        else:
            dbswitch_current = None
            current_key = None
            dictionary_fieldnames = ["db_switch", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        dbswitch_current = None
        current_key = None

    for switch_loss in np.arange(start=0.5, stop=6, step=0.25):
        if dbswitch_current != None:
            if frac_switch != dbswitch_current:
                continue
            else:
                dbswitch_current = None

        hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
            hot_bob_capacity_file=f"11_nodes_mesh_topology_35_hot_capacity_switch_loss_{round(switch_loss,2)}", cold_bob_capacity_file=f"11_nodes_mesh_topology_35_cold_capacity_switch_loss_{round(switch_loss,2)}", cmin=cmin)
        for key in hot_capacity_dict.keys():
            if current_key != None:
                if current_key == key:
                    current_key = None
                    continue
                else:
                    continue
            if key not in no_solution_list:
                try:
                    sol_dict, prob, time_to_solve = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=frac_switch,
                        time_limit=5e3, early_stop=0.003)
                except:
                    no_solution_list.append(key)
                    continue
                if optimisation_switched.check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if switch_loss == 1:
                        objective_value_at_1_dB[key] = objective_value
                    if switch_loss not in objective_values_switch_loss.keys():
                        objective_values_switch_loss[switch_loss] = {key: objective_value}
                    else:
                        objective_values_switch_loss[switch_loss][key] = objective_value

                    if data_storage_location_keep_each_loop != None:
                        dictionary = [
                            {"db_switch": round(switch_loss,2), "Graph key": key, "objective_value": objective_value}]
                        dictionary_fieldnames = ["db_switch", "Graph key", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
                    no_soln_set = []
    if data_storage_location_keep_each_loop_no_switch != None:
        if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            dictionary_fieldnames = ["Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None
    hot_capacity_dict_no_switching, cold_capacity_dict_no_switching, required_connections_no_switching, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=f"11_nodes_mesh_topology_35_hot_capacity_switch_loss_no_switch",
        cold_bob_capacity_file=f"11_nodes_mesh_topology_35_cold_capacity_switch_loss_no_switch", cmin=cmin)
    no_solution_list =[]
    objective_value_no_switch = {}
    for key in hot_capacity_dict_no_switching.keys():
        if current_key != None:
            if current_key == key:
                current_key = None
                continue
            else:
                continue
        if key not in no_solution_list:
            try:
                sol_dict, prob = optimisation_no_switching.initial_optimisation(hot_key_dict=hot_capacity_dict_no_switching[key],
                                                               cold_key_dict=cold_capacity_dict_no_switching[key],
                                                               required_connections=required_connections_no_switching[
                                                                   key], cost_det_h=cost_det_h,
                                                               cost_det_c=cost_det_c, cost_on_h=cost_on_h,
                                                               cost_on_c=cost_on_c, N=N, M=M)
            except:
                no_solution_list.append(key)
                continue
            if optimisation_switched.check_solvable(prob):
                objective_value = prob.solution.get_objective_value()
                objective_value_no_switch[key] = objective_value
                if data_storage_location_keep_each_loop_no_switch != None:
                    dictionary = [
                        {"Graph key": key, "objective_value": objective_value}]
                    dictionary_fieldnames = ["Graph key", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
                        with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["db_switch"] not in objective_values_switch_loss.keys():
                objective_values_switch_loss[row["db_switch"]] = {row["Graph key"] :row["objective_value"]}
            else:
                objective_values_switch_loss[row["db_switch"]][row["Graph key"]] = row["objective_value"]
        if data_storage_location_keep_each_loop_no_switch != None:
            objective_values_no_switch = {}
            plot_information_no_switching = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            for index, row in plot_information_no_switching.iterrows():
                if row["Graph key"] not in objective_values_no_switch.keys():
                    objective_values_no_switch[row["Graph key"]] = [row["objective_value"]]
                else:
                    objective_values_no_switch[row["Graph key"]].append(row["objective_value"])
            objective_values = {}
            for db_switch in objective_values_switch_loss.keys():
                for key in objective_values_switch_loss[frac_switch].keys():
                    if db_switch not in objective_values.keys():
                        objective_values[db_switch] = [objective_values_switch_loss[db_switch][key] / objective_values_no_switch[key]]
                    else:
                        objective_values[db_switch].append(objective_values_switch_loss[db_switch][key] / objective_values_no_switch[key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", label = "Normalised Cost of Network")
    plt.axhline(y=1, color='b', linestyle='-', label="Cost of Network without Switching")
    plt.legend()
    plt.xlabel("Switch dB Loss", fontsize=10)
    plt.ylabel("Cost of Network/Cost of Network Without Switching", fontsize=10)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("switch_loss_mesh_topology_single_graph_6.png")
    plt.show()



def cost_on_ratio_comparison(cost_det_h, cost_det_c, cost_on_h,
                                      cost_on_c, N, M, frac_switch, cmin=5000):
    objective_values_switch_loss = {}
    objective_value_at_1_dB = {}
    no_solution_list = []
    for switch_loss in np.arange(start=1, stop=6.5, step=0.5):
        hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
            hot_bob_capacity_file=f"6_nodes_mesh_topology_35_hot_capacity_switch_loss_{switch_loss}", cold_bob_capacity_file=f"6_nodes_mesh_topology_35_cold_capacity_switch_loss_{switch_loss}", cmin=cmin)
        for key in hot_capacity_dict.keys():
            if key not in no_solution_list and key == 23:
                try:
                    sol_dict, prob, time_to_solve = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=frac_switch,
                        time_limit=5e3, early_stop=0.003)
                except:
                    no_solution_list.append(key)
                    continue
                if optimisation_switched.check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if switch_loss == 1:
                        objective_value_at_1_dB[key] = objective_value
                    if switch_loss not in objective_values_switch_loss.keys():
                        objective_values_switch_loss[switch_loss] = [objective_value / objective_value_at_1_dB[key]]
                    else:
                        objective_values_switch_loss[switch_loss].append(objective_value/ objective_value_at_1_dB[key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values_switch_loss.keys():
        mean_objectives[key] = np.mean(objective_values_switch_loss[key])
        std_objectives[key] = np.std(objective_values_switch_loss[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r")
    plt.xlabel("Switch dB Loss", fontsize=10)
    plt.ylabel("Cost of Network/Cost of Network at 1dB", fontsize=10)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("switch_loss_mesh_topology_single_graph_4.png")
    plt.show()


def compare_different_detector_parameter(cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, frac_switch, cmin=5000):
    objective_values = {}
    no_solution_list = []
    for eff_hot in np.arange(start=10, stop=30, step=5):
        for eff_cold in np.arange(start = 70, stop = 100, step = 10):
            if eff_hot != 15 and eff_cold != 80:
                hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
                        hot_bob_capacity_file=f"8_nodes_mesh_topology_35_hot_capacity_{eff_cold}_eff_{eff_hot}_eff", cold_bob_capacity_file=f"8_nodes_mesh_topology_35_cold_capacity_{eff_cold}_eff_{eff_hot}_eff", cmin=cmin)
            elif eff_hot != 15:
                hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
                    hot_bob_capacity_file=f"8_nodes_mesh_topology_35_hot_capacity_{eff_hot}_eff",
                    cold_bob_capacity_file=f"8_nodes_mesh_topology_35_cold_capacity_{eff_hot}_eff",
                    cmin=cmin)
            elif eff_cold != 80:
                hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
                    hot_bob_capacity_file=f"8_nodes_mesh_topology_35_hot_capacity_{eff_cold}_eff",
                    cold_bob_capacity_file=f"8_nodes_mesh_topology_35_cold_capacity_{eff_cold}_eff",
                    cmin=cmin)
            else:
                hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
                    hot_bob_capacity_file=f"8_nodes_mesh_topology_35_hot_capacity",
                    cold_bob_capacity_file=f"8_nodes_mesh_topology_35_cold_capacity",
                    cmin=cmin)
            for key in hot_capacity_dict.keys():
                if key not in no_solution_list:
                    try:
                        sol_dict, prob, time_to_solve = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                            hot_key_dict=hot_capacity_dict[key],
                            cold_key_dict=cold_capacity_dict[key],
                           required_connections=required_connections[key],
                            cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                            cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                            fract_switch=frac_switch,
                            time_limit=5e3, early_stop=0.003)
                    except:
                        no_solution_list.append(key)
                        continue
                    if optimisation_switched.check_solvable(prob):
                        objective_value = prob.solution.get_objective_value()
                        if (eff_hot, eff_cold) not in objective_values.keys():
                            objective_values[(eff_hot, eff_cold)] = {key: objective_value}
                        else:
                            objective_values[(eff_hot, eff_cold)][key] = objective_value
    objective_value_ratios = {}
    for (eff_hot, eff_cold) in objective_values:
        for key in objective_values[(eff_hot, eff_cold)]:
            if (eff_hot, eff_cold) not in objective_value_ratios.keys():
                objective_value_ratios[(eff_hot, eff_cold)] = [objective_values[(eff_hot, eff_cold)][key] / objective_values[(15, 80)][key]]
    mean_objectives = {}
    std_objectives = {}
    for key in objective_value_ratios.keys():
        mean_objectives[key] = np.mean(objective_value_ratios[key])
        std_objectives[key] = np.std(objective_value_ratios[key])
    #### Need to figure out what we want to do here.....
    for key in mean_objectives.keys():
        print(f"The solution ratio for {key} is {mean_objectives[key]}. The standard deviation is {std_objectives[key]}")



def f_switch_parameter_sweep(hot_bob_capacity_file, cold_bob_capacity_file, hot_bob_capacity_file_no_switching, cold_bob_capacity_file_no_switching, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, cmin=5000, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_no_switch = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file,
        cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    hot_capacity_dict_no_switching, cold_capacity_dict_no_switching, required_connections_no_switching, distances_no_switching = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file_no_switching,
        cold_bob_capacity_file=cold_bob_capacity_file_no_switching, cmin=cmin)

    # graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["f_switch"]
            dataframe_of_fswitch_done = plot_information[plot_information["f_switch"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            fswitch_current = last_ratio_done.iloc[0]
        else:
            fswitch_current = None
            current_key = None
            dictionary_fieldnames = ["f_switch", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        fswitch_current = None
        current_key = None

    no_soln_set = []
    objective_value_at_frac_switch_01 = {}
    objective_values_frac_switch= {}
    for frac_switch in np.arange(start=0.0, stop=0.92, step=0.02):
        if fswitch_current != None:
            if frac_switch != fswitch_current:
                continue
            else:
                fswitch_current = None
        for key in hot_capacity_dict.keys():
            if current_key != None:
                if current_key == key:
                    current_key = None
                    continue
                else:
                    continue
            if key not in no_soln_set:
                try:
                    sol_dict, prob, time_to_solve = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=frac_switch,
                        time_limit=5e3, early_stop=0.003)
                except:
                    no_soln_set.append(key)
                    continue
                if optimisation_switched.check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if abs(frac_switch - 0.1) < 0.0001:
                        objective_value_at_frac_switch_01[key] = objective_value
                    if frac_switch not in objective_values_frac_switch.keys():
                        objective_values_frac_switch[frac_switch] = [(objective_value,key)]
                    else:
                        objective_values_frac_switch[frac_switch].append((objective_value,key))
                if data_storage_location_keep_each_loop != None:
                    dictionary = [
                        {"f_switch": frac_switch, "Graph key": key, "objective_value": objective_value}]
                    dictionary_fieldnames = ["f_switch", "Graph key", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
    no_soln_set = []
    if data_storage_location_keep_each_loop_no_switch != None:
        if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            dictionary_fieldnames = ["Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None

    for key in hot_capacity_dict_no_switching.keys():
        if current_key != None:
            if current_key == key:
                current_key = None
                continue
            else:
                continue
        if key not in no_soln_set:
            try:
                sol_dict, prob = optimisation_no_switching.initial_optimisation(hot_key_dict = hot_capacity_dict_no_switching[key], cold_key_dict = cold_capacity_dict_no_switching[key],
                                                    required_connections = required_connections_no_switching[key], cost_det_h = cost_det_h,
                                                    cost_det_c = cost_det_c, cost_on_h = cost_on_h, cost_on_c = cost_on_c, N = N, M = M)
            except:
                no_soln_set.append(key)
                continue
            if optimisation_switched.check_solvable(prob):
                objective_value = prob.solution.get_objective_value()
            if data_storage_location_keep_each_loop_no_switch != None:
                dictionary = [
                    {"Graph key": key, "objective_value": objective_value}]
                dictionary_fieldnames = ["Graph key", "objective_value"]
                if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
                    with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["f_switch"] not in objective_values_frac_switch.keys():
                objective_values_frac_switch[row["f_switch"]] = {row["Graph key"] :row["objective_value"]}
            else:
                objective_values_frac_switch[row["f_switch"]][row["Graph key"]] = row["objective_value"]
        objective_value_at_frac_switch_01 = {}
        for key in objective_values_frac_switch[0.1]:
            objective_value_at_frac_switch_01[key] = objective_values_frac_switch[0.1][key]
        if data_storage_location_keep_each_loop_no_switch != None:
            objective_values_no_switch = {}
            plot_information_no_switching = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            for index, row in plot_information_no_switching.iterrows():
                if row["Graph key"] not in objective_values_no_switch.keys():
                    objective_values_no_switch[row["Graph key"]] = [row["objective_value"]]
                else:
                    objective_values_no_switch[row["Graph key"]].append(row["objective_value"])
            objective_values = {}
            for frac_switch in objective_values_frac_switch.keys():
                for key in objective_values_frac_switch[frac_switch].keys():
                    if frac_switch not in objective_values.keys():
                        objective_values[frac_switch] = [objective_values_frac_switch[frac_switch][key] / objective_values_no_switch[key]]
                    else:
                        objective_values[frac_switch].append(objective_values_frac_switch[frac_switch][key] / objective_values_no_switch[key])
            mean_objectives = {}
            std_objectives = {}
            for key in objective_values.keys():
                mean_objectives[key] = np.mean([objective_values[key][12]])
                std_objectives[key] = np.std([objective_values[key][12]])
            mean_differences = []
            std_differences = []
            # topologies
            x = []
            for key in mean_objectives.keys():
                mean_differences.append(mean_objectives[key])
                std_differences.append(std_objectives[key])
                x.append(key)
            plt.errorbar(x, mean_differences, yerr=std_differences, color="r", capsize= 0, label = "Normalised Cost of Network")
            plt.axhline(y=1, color='b', linestyle='-', label = "Cost of Network without Switching")
            plt.legend()
            plt.xlabel("Fraction of time calibrating", fontsize=10)
            plt.ylabel("Cost of Network/Cost of Network without Switching", fontsize=10)
            # plt.legend(loc='upper right', fontsize='medium')
            plt.savefig("frac_switch_mesh_topology_single_graph_one_graph_9.png")
            plt.show()
        else:
            objective_values = {}
            for frac_switch in objective_values_frac_switch.keys():
                for key in objective_values_frac_switch[frac_switch]:
                    if frac_switch not in objective_values.keys():
                        objective_values[frac_switch] = [objective_values_frac_switch[frac_switch][key] / objective_value_at_frac_switch_01[key]]
                    else:
                        objective_values[frac_switch].append(objective_values_frac_switch[frac_switch][key] / objective_value_at_frac_switch_01[key])
            mean_objectives = {}
            std_objectives = {}
            for key in objective_values.keys():
                mean_objectives[key] = np.mean(objective_values[key])
                std_objectives[key] = np.std(objective_values[key])
            mean_differences = []
            std_differences = []
            # topologies
            x = []
            for key in mean_objectives.keys():
                mean_differences.append(mean_objectives[key])
                std_differences.append(std_objectives[key])
                x.append(key)
            plt.errorbar(x, mean_differences, yerr=std_differences, color="r", capsize= 0)
            plt.xlabel("Fraction of time calibrating", fontsize=10)
            plt.ylabel("Cost of Network/Cost of Network at 0.1 Fraction of Time Calibrating" , fontsize=10)
            # plt.legend(loc='upper right', fontsize='medium')
            plt.savefig("frac_switch_mesh_topology_single_graph_one_graph_2.png")
            plt.show()

def cmin_parameter_sweep(hot_bob_capacity_file, cold_bob_capacity_file, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, f_switch, data_storage_location_keep_each_loop = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(hot_bob_capacity_file=hot_bob_capacity_file,
                                                       cold_bob_capacity_file=cold_bob_capacity_file, cmin=1000)
    # graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["cmin"]
            dataframe_of_cmin_done = plot_information[plot_information["cmin"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            cmin_current = last_ratio_done.iloc[0]
        else:
            cmin_current = None
            current_key = None
            dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        cmin_current = None
        current_key = None
    no_soln_set = []
    objective_value_at_cmin_1000 = {}
    objective_values_cmin= {}
    lambda_values = {}
    for cmin in np.arange(start=100, stop=5000, step=100):
        if cmin_current != None:
            if cmin != cmin_current:
                continue
            else:
                cmin_current = None
        for key in hot_capacity_dict.keys():
            if key not in no_soln_set and key == 3:
                if current_key != key and current_key != None:
                    continue
                elif current_key == key:
                    current_key = None
                try:
                    for req in required_connections[key].keys():
                        required_connections[key][req] = float(cmin)
                    sol_dict, prob, time_to_solve = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=f_switch,
                        time_limit=5e3, early_stop=0.003)
                except:
                    no_soln_set.append(key)
                    continue
                if optimisation_switched.check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if abs(cmin - 1000) < 0.0001:
                        objective_value_at_cmin_1000[key] = objective_value
                    if cmin not in objective_values_cmin.keys():
                        objective_values_cmin[cmin] = [(objective_value,key)]
                    else:
                        objective_values_cmin[cmin].append((objective_value,key))
                    # print results for on nodes:
                    use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = split_sol_dict(sol_dict)
                    for lambda_key in lambda_dict.keys():
                        if lambda_key not in lambda_values.keys():
                            lambda_values[lambda_key] = {cmin: lambda_dict[lambda_key]}
                        else:
                            lambda_values[lambda_key][cmin] = lambda_dict[lambda_key]
                    print("Results for cmin:" + str(cmin))
                    for value in lambda_dict:
                        print("Value for " + str(value) + ": " + str(lambda_dict[value]))
                    for binary in binary_dict:
                        print("Value for " + str(binary) + ": " + str(binary_dict[binary]))
                    if data_storage_location_keep_each_loop != None:
                        dictionary = [
                            {"cmin": cmin, "Graph key": key, "objective_value": objective_value}]
                        dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["cmin"] not in objective_values_cmin.keys():
                objective_values_cmin[row["cmin"]] = [(row["objective_value"], row["Graph key"])]
            else:
                objective_values_cmin[row["cmin"]].append((row["objective_value"], row["Graph key"]))
        objective_value_at_cmin_1000 = {}
        for obj_value, key in objective_values_cmin[1000.0]:
            objective_value_at_cmin_1000[key] = obj_value
    objective_values_cmin = dict(sorted(objective_values_cmin.items()))
    objective_values = {}
    for cmin in objective_values_cmin.keys():
        for objective_value, key in objective_values_cmin[cmin]:
            if key in objective_value_at_cmin_1000.keys():
                if cmin not in objective_values.keys():
                    objective_values[cmin] = [objective_value / objective_value_at_cmin_1000[key]]
                else:
                    objective_values[cmin].append(objective_value / objective_value_at_cmin_1000[key])

    for key in lambda_values.keys():
        plt.plot(list(lambda_values[key].keys()), list(lambda_values[key].values()))
    plt.xlabel("Minimum Capacity Necessary (cmin)", fontsize=10)
    plt.ylabel("Number of Detectors On Site", fontsize=10)
    plt.savefig("detector_nodes_cmin_variations")
    plt.show()
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    functions = {}
    y_predictions = {}
    try:
        popt, pcov = curve_fit(exponential_fit, x, mean_differences, p0= [1,1,0], maxfev=10000000)
        y_exponential = [exponential_fit(a, popt[0], popt[1], popt[2]) for a in x]
        functions[exponential_fit] = 3
        y_predictions[exponential_fit] =y_exponential
    except:
        print("Exponential Solution Not Found")

    try:
        popt_exp, pcov_exp = curve_fit(exponential_fit_different_exponent, x, mean_differences, p0=[1, 1, 2, 0], maxfev=10000000)
        y_exponential_not_fixed_exponent = [
            exponential_fit_different_exponent(a, popt_exp[0], popt_exp[1], popt_exp[2], popt_exp[3]) for a in x]
        functions[exponential_fit_different_exponent] = 4
        y_predictions[exponential_fit_different_exponent] = y_exponential_not_fixed_exponent
    except:
        print("Exponential Solution Not Found")

    try:
        popt_lin, pcov_lin = curve_fit(linear, x, mean_differences, maxfev=10000000)
        y_lin = [linear(a, popt_lin[0], popt_lin[1]) for a in x]
        functions[linear] = 2
        y_predictions[linear] = y_lin
    except:
        print("Linear Fit Not Found")

    # fit of polynomial
    try:
        popt_quad, pcov_quad = curve_fit(quadratic, x, mean_differences, maxfev=10000000)
        y_quad = [quadratic(a, popt_quad[0], popt_quad[1], popt_quad[2]) for a in x]
        functions[quadratic] = 3
        y_predictions[quadratic] = y_quad
    except:
        print("Quadratic Fit Not Found")

    try:
        popt_cub, pcov_cub = curve_fit(cubic, x, mean_differences, maxfev=10000000)
        y_cub = [cubic(a, popt_cub[0], popt_cub[1], popt_cub[2], popt_cub[3]) for a in x]
        functions[cubic] = 4
        y_predictions[cubic] = y_cub
    except:
        print("Cubic Fit Not Found")

    try:
        popt_quar, pcov_quar = curve_fit(quartic, x, mean_differences, maxfev=10000000)
        y_quar = [quartic(a, popt_quar[0], popt_quar[1], popt_quar[2], popt_quar[3], popt_quar[4]) for a in x]
        functions[quartic] = 5
        y_predictions[quartic] = y_quar
    except:
        print("Quartic Fit Not Found")
    try:
        popt_quin, pcov_quin = curve_fit(quintic, x, mean_differences, maxfev=10000000)
        y_quin = [quintic(a, popt_quin[0], popt_quin[1], popt_quin[2], popt_quin[3], popt_quin[4], popt_quin[5]) for a in x]
        functions[quintic] = 6
        y_predictions[quintic] = y_quin
    except:
        print("Quintic Fit Not Found")

    try:
        popt_pow, pcov_pow = curve_fit(powerfit, x, mean_differences, maxfev=10000000)
        y_pow = [powerfit(a, popt_pow[0], popt_pow[1], popt_pow[2]) for a in x]
        functions[powerfit] = 3
        y_predictions[powerfit] = y_pow
    except:
        print("Powerfit Fit Not Found")

    try:
        popt_pow_2, pcov_pow_2 = curve_fit(double_power_fit, x, mean_differences, maxfev=10000000)
        y_pow_2 = [double_power_fit(a, popt_pow_2[0], popt_pow_2[1], popt_pow_2[2], popt_pow_2[3], popt_pow_2[4]) for a in x]
        functions[double_power_fit] = 5
        y_predictions[double_power_fit] = y_pow_2
    except:
        print("Double Powerfit Fit Not Found")

    try:
        popt_pow_3, pcov_pow_3 = curve_fit(triple_power_fit, x, mean_differences, maxfev=100000000)
        y_pow_3 = [double_power_fit(a, popt_pow_3[0], popt_pow_3[1], popt_pow_3[2], popt_pow_3[3], popt_pow_3[4], popt_pow_3[5], popt_pow_3[6]) for a
                   in x]
        functions[double_power_fit] = 7
        y_predictions[double_power_fit] = y_pow_3
    except:
        print("Triple Powerfit Fit Not Found")

    SSE_values = {}
    AIC_values = {}
    BIC_values = {}
    cross_val_scores = {}
    for fun in functions.keys():
        SSE_values[fun] = calculate_SSE(y_values = mean_differences, y_predictions = y_predictions[fun], m_variables = functions[fun])
        residual_plot(y_values =mean_differences, y_predictions  = y_predictions[fun], x_values = x, labels = ["Minimum Capacity Necessary (cmin)", getsource(fun).split(" ")[1].split("(")[0]])
        AIC_values[fun] = calculate_Akaike_information_criterion(y_values=mean_differences, y_predictions=y_predictions[fun], m_variables=functions[fun])
        BIC_values[fun] = calculate_BIC_values(y_values =mean_differences, y_predictions = y_predictions[fun], m_variables= functions[fun])
        # cross_val_scores[fun]= cross_val_score(y_values = mean_differences, x_values = x, k = 5, model = getsource(fun).split(" ")[1].split("(")[0]])
    for fun in AIC_values.keys():
        print("AIC values for " + str(getsource(fun).split(" ")[1].split("(")[0]) + ":" + str(AIC_values[fun]))
    for fun in BIC_values.keys():
        print("BIC values for " + str(getsource(fun).split(" ")[1].split("(")[0]) + ":" + str(BIC_values[fun]))
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", label = "Raw Data")
    # plt.plot(x, y_quar, color = "g", label = "$y = {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x + {{{}}}$".format("{:0.1e}".format(popt_quar[0]), 4, "{:0.1e}".format(popt_quar[1]), 3, "{:0.1e}".format(popt_quar[2]), 2, "{:0.1e}".format(popt_quar[3]), "{:0.1e}".format(popt_quar[4])))
    # plt.plot(x, y_quin, color="b",
    #          label="$y = {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x + {{{}}}$".format("{:0.1e}".format(popt_quin[0]), 5, "{:0.1e}".format(popt_quin[1]), 4, "{:0.1e}".format(popt_quin[2]), 3, "{:0.1e}".format(popt_quin[3]),
    #              2, "{:0.1e}".format(popt_quin[4]), "{:0.1e}".format(popt_quin[5])))
    # plt.plot(x, y_pow, color="m",
    #          label="$y = {{{}}}x^{{{}}} + {{{}}}$".format(
    #              "{:0.1e}".format(popt_pow[0]), "{:0.1e}".format(popt_pow[1]), "{:0.1e}".format(popt_pow[2])))
    # plt.plot(x, y_pow_2, color="k",
    #          label="$y = {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}$".format(
    #              "{:0.1e}".format(popt_pow_2[0]), "{:0.1e}".format(popt_pow_2[1]), "{:0.1e}".format(popt_pow_2[2]), "{:0.1e}".format(popt_pow_2[3]), "{:0.1e}".format(popt_pow_2[4])))
    plt.xlabel("Minimum Capacity Necessary (cmin)", fontsize=10)
    plt.ylabel("Cost of Network/Cost of Network at cmin = 1000" , fontsize=10)
    # plt.legend(loc='upper right', fontsize='small')
    plt.savefig("cmin_mesh_topology_single_graph_single_graph_5.png")
    plt.show()
    log_x = np.log10(x)
    log_mean = np.log10(mean_differences)
    functions = {}
    y_predictions = {}
    try:
        popt_lin, pcov_lin = curve_fit(linear, log_x, log_mean)
        y_lin = [linear(a, popt_lin[0], popt_lin[1]) for a in log_x]
        functions[linear] = 2
        y_predictions[linear] = y_lin
    except:
        print("Linear Fit Not Found")
    try:
        popt_lin_upper_limits, pcov_lin_upper_limits = curve_fit(linear, log_x[len(log_x) - 10:], log_mean[len(log_x) - 10:])
        y_lin = [linear(a, popt_lin_upper_limits[0], popt_lin_upper_limits[1]) for a in log_x]
        functions[linear] = 2
        y_predictions[linear] = y_lin
    except:
        print("Linear Fit Not Found")
        # fit of polynomial
    try:
        popt_quad, pcov_quad = curve_fit(quadratic, log_x, log_mean)
        y_quad = [quadratic(a, popt_quad[0], popt_quad[1], popt_quad[2]) for a in log_x]
        functions[quadratic] = 3
        y_predictions[quadratic] = y_quad
    except:
        print("Quadratic Fit Not Found")
    try:
        popt_pow, pcov_pow = curve_fit(powerfit, log_x, log_mean)
        y_pow = [powerfit(a, popt_pow[0], popt_pow[1], popt_pow[2]) for a in log_x]
        functions[powerfit] = 3
        y_predictions[powerfit] = y_pow
    except:
        print("Quadratic Fit Not Found")
    try:
        popt_pow_2, pcov_pow_2 = curve_fit(double_power_fit, log_x, log_mean)
        y_pow = [double_power_fit(a, popt_pow_2[0], popt_pow_2[1], popt_pow_2[2], popt_pow_2[3], popt_pow_2[4]) for a in log_x]
        functions[double_power_fit] = 5
        y_predictions[double_power_fit] = y_pow
    except:
        print("Quadratic Fit Not Found")

    SSE_values = {}
    AIC_values = {}
    cross_val_scores = {}
    for fun in functions.keys():
        SSE_values[fun] = calculate_SSE(y_values=log_mean, y_predictions=y_predictions[fun],
                                        m_variables=functions[fun])
        residual_plot(y_values=log_mean, y_predictions=y_predictions[fun], x_values=x,
                      labels=["Minimum Capacity Necessary (cmin)", getsource(fun).split(" ")[1].split("(")[0]])
        AIC_values[fun] = calculate_Akaike_information_criterion(y_values=log_mean,
                                                                 y_predictions=y_predictions[fun],
                                                                 m_variables=functions[fun])
    for fun in AIC_values.keys():
        print("log AIC values for " + str(getsource(fun).split(" ")[1].split("(")[0]) + ":" + str(AIC_values[fun]))
    plt.loglog(x, mean_differences, basex = 10, basey = 10)
    plt.show()



def cmin_parameter_sweep_with_no_switching_comaprison(hot_bob_capacity_file, cold_bob_capacity_file, hot_bob_capacity_no_switching, cold_bob_capacity_no_switching, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, f_switch, data_storage_location_keep_each_loop = None, data_storage_location_no_switching = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(hot_bob_capacity_file=hot_bob_capacity_file,
                                                       cold_bob_capacity_file=cold_bob_capacity_file, cmin=1000)
    # graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["cmin"]
            dataframe_of_cmin_done = plot_information[plot_information["cmin"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            cmin_current = last_ratio_done.iloc[0]
        else:
            cmin_current = None
            current_key = None
            dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        cmin_current = None
        current_key = None
    no_soln_set = []
    objective_value_at_cmin_1000 = {}
    objective_values_cmin= {}
    lambda_values = {}
    for cmin in np.arange(start=100, stop=5000, step=100):
        if cmin_current != None:
            if cmin != cmin_current:
                continue
            else:
                cmin_current = None
        for key in hot_capacity_dict.keys():
            if key not in no_soln_set:
                if current_key != key and current_key != None:
                    continue
                elif current_key == key:
                    current_key = None
                try:
                    for req in required_connections[key].keys():
                        required_connections[key][req] = float(cmin)
                    sol_dict, prob, time_to_solve = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=f_switch,
                        time_limit=5e3, early_stop=0.003)
                except:
                    no_soln_set.append(key)
                    continue
                if optimisation_switched.check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if abs(cmin - 1000) < 0.0001:
                        objective_value_at_cmin_1000[key] = objective_value
                    if cmin not in objective_values_cmin.keys():
                        objective_values_cmin[cmin] = [(objective_value,key)]
                    else:
                        objective_values_cmin[cmin].append((objective_value,key))
                    # print results for on nodes:
                    use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = split_sol_dict(sol_dict)
                    for lambda_key in lambda_dict.keys():
                        if lambda_key not in lambda_values.keys():
                            lambda_values[lambda_key] = {cmin: lambda_dict[lambda_key]}
                        else:
                            lambda_values[lambda_key][cmin] = lambda_dict[lambda_key]
                    print("Results for cmin:" + str(cmin))
                    for value in lambda_dict:
                        print("Value for " + str(value) + ": " + str(lambda_dict[value]))
                    for binary in binary_dict:
                        print("Value for " + str(binary) + ": " + str(binary_dict[binary]))
                    if data_storage_location_keep_each_loop != None:
                        dictionary = [
                            {"cmin": cmin, "Graph key": key, "objective_value": objective_value}]
                        dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)

    hot_capacity_dict_no_switch, cold_capacity_dict_no_switch, required_connections_no_switch, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_no_switching,
        cold_bob_capacity_file=cold_bob_capacity_no_switching, cmin=1000)
    # graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    if data_storage_location_no_switching != None:
        if os.path.isfile(data_storage_location_no_switching + '.csv'):
            plot_information = pd.read_csv(data_storage_location_no_switching + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["cmin"]
            dataframe_of_cmin_done = plot_information[plot_information["cmin"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            cmin_current = last_ratio_done.iloc[0]
        else:
            cmin_current = None
            current_key = None
            dictionary_fieldnames = ["cmin","Graph key", "objective_value"]
            with open(data_storage_location_no_switching + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        cmin_current = None
        current_key = None
    no_soln_set = []
    objective_values_cmin_no_switching = {}
    lambda_values = {}
    for cmin in np.arange(start=100, stop=5000, step=100):
        if cmin_current != None:
            if cmin != cmin_current:
                continue
            else:
                cmin_current = None
        for key in hot_capacity_dict_no_switch.keys():
            if key not in no_soln_set and key != 3:
                if current_key != key and current_key != None:
                    continue
                elif current_key == key:
                    current_key = None
                try:
                    for req in required_connections_no_switch[key].keys():
                        required_connections_no_switch[key][req] = float(cmin)
                    sol_dict, prob = optimisation_no_switching.initial_optimisation(
                        hot_key_dict=hot_capacity_dict_no_switch[key],
                        cold_key_dict=cold_capacity_dict_no_switch[key],
                        required_connections=required_connections_no_switch[key], cost_det_h=cost_det_h,
                        cost_det_c=cost_det_c, cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M)
                except:
                    no_soln_set.append(key)
                    continue
                if optimisation_switched.check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if abs(cmin - 1000) < 0.0001:
                        objective_value_at_cmin_1000[key] = objective_value
                    if cmin not in objective_values_cmin_no_switching.keys():
                        objective_values_cmin_no_switching[cmin] = [(objective_value, key)]
                    else:
                        objective_values_cmin_no_switching[cmin].append((objective_value, key))
                    # print results for on nodes:
                    use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = split_sol_dict(sol_dict)
                    for lambda_key in lambda_dict.keys():
                        if lambda_key not in lambda_values.keys():
                            lambda_values[lambda_key] = {cmin: lambda_dict[lambda_key]}
                        else:
                            lambda_values[lambda_key][cmin] = lambda_dict[lambda_key]
                    print("Results for cmin:" + str(cmin))
                    for value in lambda_dict:
                        print("Value for " + str(value) + ": " + str(lambda_dict[value]))
                    for binary in binary_dict:
                        print("Value for " + str(binary) + ": " + str(binary_dict[binary]))
                    if data_storage_location_no_switching != None:
                        dictionary = [
                            {"cmin": cmin, "Graph key": key, "objective_value": objective_value}]
                        dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
                        if os.path.isfile(data_storage_location_no_switching + '.csv'):
                            with open(data_storage_location_no_switching + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_no_switching + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)

    if data_storage_location_no_switching != None:
        plot_information = pd.read_csv(data_storage_location_no_switching + ".csv")
        for index, row in plot_information.iterrows():
            if row["cmin"] not in objective_values_cmin_no_switching.keys():
                objective_values_cmin_no_switching[row["cmin"]] = [(row["objective_value"], row["Graph key"])]
            else:
                objective_values_cmin_no_switching[row["cmin"]].append((row["objective_value"], row["Graph key"]))
        objective_value_at_cmin_1000 = {}
        for obj_value, key in objective_values_cmin_no_switching[1000.0]:
            objective_value_at_cmin_1000[key] = obj_value
    objective_values_cmin = dict(sorted(objective_values_cmin.items()))
    objective_values_cmin_no_switching = dict(sorted(objective_values_cmin_no_switching.items()))
    objective_values = {}
    for cmin in objective_values_cmin.keys():
        for objective_value, key in objective_values_cmin[cmin]:
            for objective_value_no_switch, key_no_switch in objective_values_cmin_no_switching[cmin]:
                if key == key_no_switch:
                    if cmin not in objective_values.keys():
                        objective_values[cmin] = [objective_value / objective_value_no_switch]
                    else:
                        objective_values[cmin].append(objective_value / objective_value_no_switch)

    for key in lambda_values.keys():
        plt.plot(list(lambda_values[key].keys()), list(lambda_values[key].values()))
    plt.xlabel("Minimum Capacity Necessary (cmin)", fontsize=10)
    plt.ylabel("Number of Detectors On Site", fontsize=10)
    plt.savefig("detector_nodes_cmin_variations")
    plt.show()
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    functions = {}
    y_predictions = {}
    try:
        popt, pcov = curve_fit(exponential_fit, x, mean_differences, p0= [1,1,0], maxfev=10000000)
        y_exponential = [exponential_fit(a, popt[0], popt[1], popt[2]) for a in x]
        functions[exponential_fit] = 3
        y_predictions[exponential_fit] =y_exponential
    except:
        print("Exponential Solution Not Found")

    try:
        popt_exp, pcov_exp = curve_fit(exponential_fit_different_exponent, x, mean_differences, p0=[1, 1, 2, 0], maxfev=10000000)
        y_exponential_not_fixed_exponent = [
            exponential_fit_different_exponent(a, popt_exp[0], popt_exp[1], popt_exp[2], popt_exp[3]) for a in x]
        functions[exponential_fit_different_exponent] = 4
        y_predictions[exponential_fit_different_exponent] = y_exponential_not_fixed_exponent
    except:
        print("Exponential Solution Not Found")

    try:
        popt_lin, pcov_lin = curve_fit(linear, x, mean_differences, maxfev=10000000)
        y_lin = [linear(a, popt_lin[0], popt_lin[1]) for a in x]
        functions[linear] = 2
        y_predictions[linear] = y_lin
    except:
        print("Linear Fit Not Found")

    # fit of polynomial
    try:
        popt_quad, pcov_quad = curve_fit(quadratic, x, mean_differences, maxfev=10000000)
        y_quad = [quadratic(a, popt_quad[0], popt_quad[1], popt_quad[2]) for a in x]
        functions[quadratic] = 3
        y_predictions[quadratic] = y_quad
    except:
        print("Quadratic Fit Not Found")

    try:
        popt_cub, pcov_cub = curve_fit(cubic, x, mean_differences, maxfev=10000000)
        y_cub = [cubic(a, popt_cub[0], popt_cub[1], popt_cub[2], popt_cub[3]) for a in x]
        functions[cubic] = 4
        y_predictions[cubic] = y_cub
    except:
        print("Cubic Fit Not Found")

    try:
        popt_quar, pcov_quar = curve_fit(quartic, x, mean_differences, maxfev=10000000)
        y_quar = [quartic(a, popt_quar[0], popt_quar[1], popt_quar[2], popt_quar[3], popt_quar[4]) for a in x]
        functions[quartic] = 5
        y_predictions[quartic] = y_quar
    except:
        print("Quartic Fit Not Found")
    try:
        popt_quin, pcov_quin = curve_fit(quintic, x, mean_differences, maxfev=10000000)
        y_quin = [quintic(a, popt_quin[0], popt_quin[1], popt_quin[2], popt_quin[3], popt_quin[4], popt_quin[5]) for a in x]
        functions[quintic] = 6
        y_predictions[quintic] = y_quin
    except:
        print("Quintic Fit Not Found")

    try:
        popt_pow, pcov_pow = curve_fit(powerfit, x, mean_differences, maxfev=10000000)
        y_pow = [powerfit(a, popt_pow[0], popt_pow[1], popt_pow[2]) for a in x]
        functions[powerfit] = 3
        y_predictions[powerfit] = y_pow
    except:
        print("Powerfit Fit Not Found")

    try:
        popt_pow_2, pcov_pow_2 = curve_fit(double_power_fit, x, mean_differences, maxfev=10000000)
        y_pow_2 = [double_power_fit(a, popt_pow_2[0], popt_pow_2[1], popt_pow_2[2], popt_pow_2[3], popt_pow_2[4]) for a in x]
        functions[double_power_fit] = 5
        y_predictions[double_power_fit] = y_pow_2
    except:
        print("Double Powerfit Fit Not Found")

    try:
        popt_pow_3, pcov_pow_3 = curve_fit(triple_power_fit, x, mean_differences, maxfev=100000000)
        y_pow_3 = [double_power_fit(a, popt_pow_3[0], popt_pow_3[1], popt_pow_3[2], popt_pow_3[3], popt_pow_3[4], popt_pow_3[5], popt_pow_3[6]) for a
                   in x]
        functions[double_power_fit] = 7
        y_predictions[double_power_fit] = y_pow_3
    except:
        print("Triple Powerfit Fit Not Found")

    SSE_values = {}
    AIC_values = {}
    BIC_values = {}
    cross_val_scores = {}
    for fun in functions.keys():
        SSE_values[fun] = calculate_SSE(y_values = mean_differences, y_predictions = y_predictions[fun], m_variables = functions[fun])
        residual_plot(y_values =mean_differences, y_predictions  = y_predictions[fun], x_values = x, labels = ["Minimum Capacity Necessary (cmin)", getsource(fun).split(" ")[1].split("(")[0]])
        AIC_values[fun] = calculate_Akaike_information_criterion(y_values=mean_differences, y_predictions=y_predictions[fun], m_variables=functions[fun])
        BIC_values[fun] = calculate_BIC_values(y_values =mean_differences, y_predictions = y_predictions[fun], m_variables= functions[fun])
        # cross_val_scores[fun]= cross_val_score(y_values = mean_differences, x_values = x, k = 5, model = getsource(fun).split(" ")[1].split("(")[0]])
    for fun in AIC_values.keys():
        print("AIC values for " + str(getsource(fun).split(" ")[1].split("(")[0]) + ":" + str(AIC_values[fun]))
    for fun in BIC_values.keys():
        print("BIC values for " + str(getsource(fun).split(" ")[1].split("(")[0]) + ":" + str(BIC_values[fun]))
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", label = "Normalised Cost")
    plt.axhline(y=1, color='b', linestyle='-', label="Cost of Network without Switching")
    # plt.plot(x, y_quar, color = "g", label = "$y = {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x + {{{}}}$".format("{:0.1e}".format(popt_quar[0]), 4, "{:0.1e}".format(popt_quar[1]), 3, "{:0.1e}".format(popt_quar[2]), 2, "{:0.1e}".format(popt_quar[3]), "{:0.1e}".format(popt_quar[4])))
    # plt.plot(x, y_quin, color="b",
    #          label="$y = {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}x + {{{}}}$".format("{:0.1e}".format(popt_quin[0]), 5, "{:0.1e}".format(popt_quin[1]), 4, "{:0.1e}".format(popt_quin[2]), 3, "{:0.1e}".format(popt_quin[3]),
    #              2, "{:0.1e}".format(popt_quin[4]), "{:0.1e}".format(popt_quin[5])))
    # plt.plot(x, y_pow, color="m",
    #          label="$y = {{{}}}x^{{{}}} + {{{}}}$".format(
    #              "{:0.1e}".format(popt_pow[0]), "{:0.1e}".format(popt_pow[1]), "{:0.1e}".format(popt_pow[2])))
    # plt.plot(x, y_pow_2, color="k",
    #          label="$y = {{{}}}x^{{{}}} + {{{}}}x^{{{}}} + {{{}}}$".format(
    #              "{:0.1e}".format(popt_pow_2[0]), "{:0.1e}".format(popt_pow_2[1]), "{:0.1e}".format(popt_pow_2[2]), "{:0.1e}".format(popt_pow_2[3]), "{:0.1e}".format(popt_pow_2[4])))
    plt.xlabel("Minimum Capacity Necessary (cmin) \s", fontsize=10)
    plt.ylabel("Cost of Network with switched/ without switches" , fontsize=10)
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig("cmin_mesh_topology_single_graph_single_graph_4.png")
    plt.show()
    log_x = np.log10(x)
    log_mean = np.log10(mean_differences)
    functions = {}
    y_predictions = {}
    try:
        popt_lin, pcov_lin = curve_fit(linear, log_x, log_mean)
        y_lin = [linear(a, popt_lin[0], popt_lin[1]) for a in log_x]
        functions[linear] = 2
        y_predictions[linear] = y_lin
    except:
        print("Linear Fit Not Found")
    try:
        popt_lin_upper_limits, pcov_lin_upper_limits = curve_fit(linear, log_x[len(log_x) - 10:], log_mean[len(log_x) - 10:])
        y_lin = [linear(a, popt_lin_upper_limits[0], popt_lin_upper_limits[1]) for a in log_x]
        functions[linear] = 2
        y_predictions[linear] = y_lin
    except:
        print("Linear Fit Not Found")
        # fit of polynomial
    try:
        popt_quad, pcov_quad = curve_fit(quadratic, log_x, log_mean)
        y_quad = [quadratic(a, popt_quad[0], popt_quad[1], popt_quad[2]) for a in log_x]
        functions[quadratic] = 3
        y_predictions[quadratic] = y_quad
    except:
        print("Quadratic Fit Not Found")
    try:
        popt_pow, pcov_pow = curve_fit(powerfit, log_x, log_mean)
        y_pow = [powerfit(a, popt_pow[0], popt_pow[1], popt_pow[2]) for a in log_x]
        functions[powerfit] = 3
        y_predictions[powerfit] = y_pow
    except:
        print("Quadratic Fit Not Found")
    try:
        popt_pow_2, pcov_pow_2 = curve_fit(double_power_fit, log_x, log_mean)
        y_pow = [double_power_fit(a, popt_pow_2[0], popt_pow_2[1], popt_pow_2[2], popt_pow_2[3], popt_pow_2[4]) for a in log_x]
        functions[double_power_fit] = 5
        y_predictions[double_power_fit] = y_pow
    except:
        print("Quadratic Fit Not Found")

    SSE_values = {}
    AIC_values = {}
    cross_val_scores = {}
    for fun in functions.keys():
        SSE_values[fun] = calculate_SSE(y_values=log_mean, y_predictions=y_predictions[fun],
                                        m_variables=functions[fun])
        residual_plot(y_values=log_mean, y_predictions=y_predictions[fun], x_values=x,
                      labels=["Minimum Capacity Necessary (cmin)", getsource(fun).split(" ")[1].split("(")[0]])
        AIC_values[fun] = calculate_Akaike_information_criterion(y_values=log_mean,
                                                                 y_predictions=y_predictions[fun],
                                                                 m_variables=functions[fun])
    for fun in AIC_values.keys():
        print("log AIC values for " + str(getsource(fun).split(" ")[1].split("(")[0]) + ":" + str(AIC_values[fun]))
    plt.loglog(x, mean_differences, basex = 10, basey = 10)
    plt.show()


def heuristic_comparison_analysis(hot_bob_capacity_file, cold_bob_capacity_file,N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, cmin, f_switch, data_storage_location_keep_each_loop = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file,
        cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    # graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["Graph key"]
            dataframe_of_cmin_done = plot_information[plot_information["Graph key"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            dictionary_fieldnames = ["Graph key", "objective_value", "heuristic_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None
    no_soln_set = []
    objective_values_dict = {}

    for key in hot_capacity_dict.keys():
        if key not in no_soln_set:
            if current_key != key and current_key != None:
                continue
            elif current_key == key:
                current_key = None
                continue
            try:
                sol_dict, prob, time_to_solve = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=f_switch,
                        time_limit=5e3, early_stop=0.0002)
            except:
                no_soln_set.append(key)
                continue
            if optimisation_switched.check_solvable(prob):
                model = LP_relaxation.LP_Switched_fixed_switching_time_relaxation(name=f"problem",
                                                                    hot_key_dict=hot_capacity_dict[key],
                                                                    cold_key_dict=cold_capacity_dict[key],
                                                                    Lambda=N,
                                                                    required_connections=required_connections[key])
                heuristic = Relaxed_heuristic.Relaxed_Heuristic(c_det_hot=cost_det_h, c_det_cold=cost_det_c, c_cold_on=cost_on_c, c_hot_on=cost_on_h, Lambda=N,
                                              f_switch=f_switch, M=M)
                t_1 = time.time()
                model_best = heuristic.full_recursion(initial_model=model)
                t_2 = time.time()
                print("Time for recursion " + str(t_2 - t_1))
                try:
                    heuristic_value = heuristic.calculate_current_solution_cost(model_best)
                    objective_value = prob.solution.get_objective_value()
                    if key not in objective_values_dict.keys():
                        objective_values_dict[key] = [(objective_value, heuristic_value)]
                    else:
                        objective_values_dict[key].append((objective_value, heuristic_value))
                    if data_storage_location_keep_each_loop != None:
                        dictionary = [
                            {"Graph key": key,"objective_value": objective_value, "heuristic_value": heuristic_value}]
                        dictionary_fieldnames =  ["Graph key", "objective_value", "heuristic_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
                except:
                    print("No solution in heuristic model")
                    continue

    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Graph key"] not in objective_values_dict.keys():
                objective_values_dict[row["Graph key"]] = [
                    (row["objective_value"], row["heuristic_value"])]
            else:
                objective_values_dict[row["Graph key"]].append(
                    (row["objective_value"], row["heuristic_value"]))
    percentage_difference_for_number_nodes = {}
    for key in objective_values_dict.keys():
        number_nodes = optimisation_switched.get_number_of_nodes(hot_capacity_dict[key], cold_capacity_dict[key])
        if number_nodes not in percentage_difference_for_number_nodes.keys():
            percentage_difference_for_number_nodes[number_nodes] = [100 * (objective_values_dict[key][i][1]- objective_values_dict[key][i][0])/objective_values_dict[key][i][0] for i in range(len(objective_values_dict[key]))]
        else:
            percentage_difference_for_number_nodes[number_nodes].extend([100 * (objective_values_dict[key][0][1]- objective_values_dict[key][i][0])/objective_values_dict[key][i][0] for i in range(len(objective_values_dict[key]))])
    mean_objectives = {}
    std_objectives = {}
    for key in percentage_difference_for_number_nodes.keys():
        mean_objectives[key] = np.mean(percentage_difference_for_number_nodes[key])
        std_objectives[key] = np.std(percentage_difference_for_number_nodes[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r")
    plt.xlabel("Number of Nodes in the Graph", fontsize=10)
    plt.ylabel("Percentage Difference of Heuristic to Optimal Solution", fontsize=10)
    plt.savefig("heuristic_quality_plot_2.png")
    plt.show()



################################## Parameter Fit Functions ################################################


def exponential_fit(x, a, exp, c):
    return a * np.exp(exp * x) + c

def exponential_fit_different_exponent(x, a, b, exp, c):
    return a * np.power(b, (exp * x)) + c

def powerfit(x, a, b, c):
    return a * np.power(x, b) + c


def linear(x, a, b):
    return a * (x) + b

def quadratic(x, a, b, c):
    return a * (x ** 2) + b * x + c

def cubic(x, a, b, c, d):
    return a * (x ** 3) + b * (x  ** 2) + c * x + d

def quartic(x, a, b, c, d, e):
    return a * (x ** 4) + b * (x  ** 3) + c * (x ** 2) + d * x  + e

def quintic(x, a, b, c, d, e, f):
    return a * (x ** 5) + b * (x  ** 4) + c * (x ** 3) + d * (x ** 2)  + e * x + f

def double_power_fit(x, a, b, c, d, e):
    return a * np.power(x, b) + c * np.power(x,d) + e

def triple_power_fit(x, a, b, c, d, e, f, g):
    return a * np.power(x, b) + c * np.power(x,d) + e * np.power(x,f) + g


def calculate_SSE(y_values, y_predictions, m_variables):
    """
    Calculate the SSE for the regression line.

    Parameters
    ----------
    y_values:  The list of the measured y values of the model
    y_predictions: The list of the predicted y values of the model at the x values corresponding to y_values
    m_variables: number of variables in the model

    Returns: SSE value
    -------

    """
    if len(y_values) !=  len(y_predictions):
        print("The size of the arrays are invalid")
        raise ValueError
    if len(y_values) < m_variables + 1:
        print("There are not enough data points")
        raise ValueError
    sum_of_square_differences = 0.0
    for i in range(len(y_values)):
        sum_of_square_differences += np.power(y_values[i] - y_predictions[i], 2)
    return sum_of_square_differences / (len(y_values) - m_variables - 1)


def residual_plot(y_values, y_predictions, x_values, labels):
    """
    Plots the graph of residuals for the regression line

    Parameters
    ----------
    y_values : The list of measured y values of the model
    y_predictions : The list of predicted y values of the model at the x values corresponding to y_values
    x_values : The x values list
    labels : [x_label, y_label unit]
    -------

    """
    if len(y_values) != len(y_predictions) or len(y_values) != len(x_values):
        print("The size of the arrays are invalid")
        raise ValueError
    residues = []
    for i in range(len(y_values)):
        residues.append(y_values[i] - y_predictions[i])
    plt.plot(x_values, residues)
    plt.xlabel(labels[0], fontsize=10)
    plt.ylabel(f"Residues of y_values / {labels[1]}", fontsize=10)
    plt.show()


def perform_cross_validation_model_fit(y_values, x_values, k, model):
    # if model[0] == "e":
    #     model = "exponential"
    #

    clf = svm.SVC(kernel = model, C= 1, random_state=42)
    scores = cross_val_score(clf, x_values, y_values, cv=k)
    return scores


def calculate_Akaike_information_criterion(y_values, y_predictions, m_variables):
    if len(y_values) != len(y_predictions):
        print("The size of the arrays are invalid")
        raise ValueError
    if len(y_values) < m_variables + 1:
        print("There are not enough data points")
        raise ValueError
    n = len(y_values)
    sum_of_square_differences = 0.0
    for i in range(len(y_values)):
        sum_of_square_differences += np.power(y_values[i] - y_predictions[i], 2)
    sum_of_square_differences = sum_of_square_differences / n
    aic = 2 * m_variables + n * np.log(sum_of_square_differences)
    if n < 40:
        aic += 2* m_variables *(m_variables + 1)/ (n-m_variables -1)
    return aic

def calculate_BIC_values(y_values, y_predictions, m_variables):
    if len(y_values) != len(y_predictions):
        print("The size of the arrays are invalid")
        raise ValueError
    if len(y_values) < m_variables + 1:
        print("There are not enough data points")
        raise ValueError
    n = len(y_values)
    sum_of_square_differences = 0.0
    for i in range(len(y_values)):
        sum_of_square_differences += np.power(y_values[i] - y_predictions[i], 2)
    sum_of_square_differences = sum_of_square_differences / n
    bic = m_variables * np.log(n) + n * np.log(sum_of_square_differences)
    return bic




def test_differences(hot_bob_capacity_file, cold_bob_capacity_file, hot_bob_capacity_no_switching, cold_bob_capacity_no_switching, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, f_switch, cmin, data_storage_location_keep_each_loop = None, data_storage_location_no_switching = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(hot_bob_capacity_file=hot_bob_capacity_file,
                                                       cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    # graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)

    no_soln_set = []
    differences = []
    for key in hot_capacity_dict.keys():
        if key not in no_soln_set:
            try:
                sol_dict, prob, time_to_solve = initial_optimisation_switching_fixed_switching_ratio(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=f_switch,
                        time_limit=5e3, early_stop=0.003)


                sol_dict_2, prob_2, time_to_solve_2 = initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=f_switch,
                        time_limit=5e3, early_stop=0.003)
                differences.append(prob_2.solution.get_objective_value() - prob.solution.get_objective_value())
            except:
                continue
    print(differences)

# For ignoring building and maintenance costs: C_{on}^{c} = 1.714, C_{det}^{c} = 3.57, C_{on}^{h} = 1, C_{det}^{h} = 3.14
# For including building and maintenance costs: C_{on}^{c} = 3.27, C_{det}^{c} = 1.136, C_{on}^{h} = 1.68, C_{det}^{h} = 1
# Each cooling chamber is capable of holding 24 detectors - 2 needed per connection so N = 12 is good here, set the
# number of different paths to M = 2.


### SPADS + Link Cost (5k *2 + 100k)
### cryostat - 40k

### so turning on : 40 + Building  fees etc.
### adding new connection : 110k


### for cold: SNSPD + Link Cost (25k + 100k)
### cryostat: 50k


### so turning on : 50 k + building fees (higher because need more for ensuring cryogenically able)
### adding new connection 125k

# If instead we assume 2 cryo-chambers are added every time a node is turned on then the values are
# For ignoring building and maintenance costs: C_{on}^{c} = 1.714, C_{det}^{c} = 1.786, C_{on}^{h} = 1, C_{det}^{h} = 1.57
# For including building and maintenance costs: C_{on}^{c} = 3.8, C_{det}^{c} = 1.14, C_{on}^{h} = 2, C_{det}^{h} = 1
# Each cooling chamber is capable of holding 24 detectors - 2 needed per connection so N = 24 is good here, set the
# number of different paths to M = 2.



if __name__ == "__main__":
    # find_critical_ratio_cooled_uncooled(hot_bob_capacity_file = "9_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file = "9_nodes_mesh_topology_35_cold_capacity", graph_edge_data_file = "9_nodes_mesh_topology_35_edge_positions",
    #                                     graph_node_data_file = "9_nodes_mesh_topology_35_node_positions", N = 12, M = 2, f_switch= 0.1, cost_det_h = 1, cost_on_h = 1.68,
    #                                     cmin=1000, data_storage_location_keep_each_loop="critical_cooling_ratio_80_eff_15_eff_mult_nodes_4", complete = True)
    # f_switch_parameter_sweep(hot_bob_capacity_file = "10_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file = "10_nodes_mesh_topology_35_cold_capacity", hot_bob_capacity_file_no_switching = "10_nodes_mesh_topology_35_hot_capacity_no_switch",
    #                          cold_bob_capacity_file_no_switching = "10_nodes_mesh_topology_35_cold_capacity_no_switch", N = 1200, M = 2, cost_det_h = 1, cost_det_c = 1.136, cost_on_h = 1.68, cost_on_c = 3.27,
    #                          cmin=1000, data_storage_location_keep_each_loop="fswitch_parameter_sweep",
    #                          data_storage_location_keep_each_loop_no_switch="fswitch_parameter_sweep_no_switch")
    # cmin_parameter_sweep_with_no_switching_comaprison(hot_bob_capacity_file = "10_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file= "10_nodes_mesh_topology_35_cold_capacity",
    #                                                   hot_bob_capacity_no_switching= "10_nodes_mesh_topology_35_hot_capacity_no_switch", cold_bob_capacity_no_switching ="10_nodes_mesh_topology_35_cold_capacity_no_switch", N = 12,
    #                                                   M = 2, cost_det_h = 1, cost_det_c = 1.136, cost_on_h = 1.68, cost_on_c = 3.27, f_switch = 0.1)
                                                      # data_storage_location_keep_each_loop="cmin_parameter_sweep",
                                                      # data_storage_location_no_switching="cmin_parameter_sweep_no_switching")
    # test_differences(hot_bob_capacity_file = "10_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file= "10_nodes_mesh_topology_35_cold_capacity",
    #                                                   hot_bob_capacity_no_switching= "10_nodes_mesh_topology_35_hot_capacity_no_switch", cold_bob_capacity_no_switching ="10_nodes_mesh_topology_35_cold_capacity_no_switch", N = 1200,
    #                                                   M = 2, cost_det_h = 1, cost_det_c = 1.136, cost_on_h = 1.68, cost_on_c = 3.27, f_switch = 0.1, cmin =1000,
    #                                                   data_storage_location_keep_each_loop="cmin_parameter_sweep",
    #                                                   data_storage_location_no_switching="cmin_parameter_sweep_no_switching")
    # find_critical_ratio_cooled_uncooled(hot_bob_capacity_file="9_nodes_mesh_topology_35_hot_capacity",
    #                                     cold_bob_capacity_file="9_nodes_mesh_topology_35_cold_capacity",
    #                                     graph_edge_data_file="9_nodes_mesh_topology_35_edge_positions",
    #                                     graph_node_data_file="9_nodes_mesh_topology_35_node_positions", N=12, M=2,
    #                                     f_switch=0.1, cost_det_h=1, cost_on_c=3.27, cost_on_h=1.68,
    #                                     cmin=1000,
    #                                     data_storage_location_keep_each_loop="critical_cooling_ratio_80_eff_15_eff_mult_nodes_4")
    # cmin_parameter_sweep(hot_bob_capacity_file = "8_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file = "8_nodes_mesh_topology_35_cold_capacity", N = 12, M = 2, cost_det_h = 1, cost_det_c = 1.136, cost_on_h = 1.68,
    #                          cost_on_c = 3.27, f_switch=0.1, data_storage_location_keep_each_loop = None)
    # heuristic_comparison_analysis(hot_bob_capacity_file = "heuristic_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file = "heuristic_nodes_mesh_topology_35_cold_capacity", N = 12, M = 2, cost_det_h = 1, cost_det_c = 1.136,
    #                               cost_on_h = 1.68, cost_on_c = 3.27, cmin = 1000, f_switch = 0.1, data_storage_location_keep_each_loop="heuristic_investigation_save_location_2")

    # f_switch_parameter_sweep(hot_bob_capacity_file = "8_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file ="8_nodes_mesh_topology_35_cold_capacity" , N = 12, M = 2, cost_det_h = 1, cost_det_c = 1.136, cost_on_h = 1.68,
    #                          cost_on_c = 3.27, cmin=1000, data_storage_location_keep_each_loop=None)

    # # compare_different_detector_parameter(cost_det_h  = 1, cost_det_c = 10, cost_on_h = 1.68, cost_on_c = 20, N = 12, M= 2, frac_switch = 0.1, cmin=1000)
    # difference_in_solution_costs_for_various_topologies(hot_bob_capacity_file_list = ["ring_topology_hot_capacity_data", "bus_topology_hot_capacity_data", "star_topology_not_centre_detector_hot_capacity_data_2", "star_topology_centre_detector_hot_capacity_data", "mesh_topology_35_hot_capacity_data_complete_final"],
    #                                                     cold_bob_capacity_file_list = ["ring_topology_cold_capacity_data", "bus_topology_cold_capacity_data", "star_topology_not_centre_detector_cold_capacity_data_2", "star_topology_centre_detector_cold_capacity_data", "mesh_topology_35_cold_capacity_data_complete_final"],
    #                                                     graph_edge_data_file_list = ["ring_topology_edge_position_data", "bus_topology_edge_position_data", "star_topology_not_centre_detector_edge_position_data_2", "star_topology_centre_detector_edge_position_data", "mesh_topology_35_edge_position_data_complete_final"],
    #                                                     graph_node_data_file_list = ["ring_topology_node_position_data", "bus_topology_node_position_data", "star_topology_not_centre_detector_node_position_data_2", "star_topology_centre_detector_node_position_data", "mesh_topology_35_node_position_data_complete_final"],
    #                                                     topologies_list = ["ring topology", "bus topology", "star topology- detector not in centre", "star topology- detector in centre", "mesh topology, average connectivity=3.5"],
    #                                                     N =24, M = 2, cost_det_h = 1, cost_det_c = 1.14, cost_on_h =2,
    #                                                     cost_on_c = 3.8, KT = 1000, cmin=5000,
    #                                                     data_storage_location_keep_each_loop="topology_comparison")
    # hot_capacity_dict_multiple_graphs, cold_capacity_dict_multiple_graphs, required_connections_multiple_graphs, distances = import_switched_network_values_multiple_graphs(hot_bob_capacity_file = "mesh_topology_75_hot_capacity_data_2", cold_bob_capacity_file = "mesh_topology_75_cold_capacity_data_2", cmin = 50000)
    # switch_loss_cost_comparison(cost_det_h = 1, cost_det_c = 1.136, cost_on_h =  1.68, cost_on_c = 3.27, N = 1200, M = 2, frac_switch=0.1, cmin=1000, data_storage_location_keep_each_loop="db_switch_loss", data_storage_location_keep_each_loop_no_switch="db_switch_no_switching")

    # time_taken_with_increasing_number_of_nodes(hot_bob_capacity_file="5_nodes_mesh_topology_35_hot_capacity",
    #                                                cold_bob_capacity_file="5_nodes_mesh_topology_35_cold_capacity", graph_edge_data_file = "5_nodes_mesh_topology_35_edge_positions",
    #                                            graph_node_data_file = "5_nodes_mesh_topology_35_node_positions", cost_det_h=1, cost_det_c=2,
    #                                                 cost_on_h=2, cost_on_c=4, N=12, M=2,
    #                                                 frac_switch=0.1, cmin=1000)

    # hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(hot_bob_capacity_file="3_nodes_mesh_topology_35_hot_capacity",
    #                                                cold_bob_capacity_file="3_nodes_mesh_topology_35_cold_capacity", cmin=1000)
    #
    # graphs = import_graph_structure(node_information = "3_nodes_mesh_topology_35_node_positions", edge_information = "3_nodes_mesh_topology_35_edge_positions")
    # for i in range(len(hot_capacity_dict)):
    #     time = {}
    #     try:
    #         sol_dict, prob, time_to_solve= initial_optimisation_switching_fixed_switching_ratio(hot_key_dict=hot_capacity_dict[i],
    #                                                 cold_key_dict=cold_capacity_dict[i],
    #                                                 required_connections=required_connections[i],
    #                                                 cost_det_h=1, cost_det_c=2,
    #                                                 cost_on_h=2, cost_on_c=4, N=12, M=2,
    #                                                 fract_switch=0.15,
    #                                                 time_limit=5e2, early_stop=0.003)
    #         q_solns, lambda_soln, w_solns, binary_solns, detector_usage_soln= split_sol_dict(sol_dict)
    #         print(lambda_soln)
    #         optimisation_switched.plot_graph(graphs[i], binary_dict=binary_solns)
    #     except:
    #         continue
    # KT_required_to_make_significant_difference(hot_bob_capacity_file = "60_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file = "60_nodes_mesh_topology_35_cold_capacity", graph_edge_data_file = "60_nodes_mesh_topology_35_edge_positions",
    #                                            graph_node_data_file = "60_nodes_mesh_topology_35_node_positions", N = 24, M = 1, cost_det_h = 1.57, cost_det_c = 1.786, cost_on_h = 1, cost_on_c = 1.714,
    #                                            cmin=10000, data_storage_location_keep_each_loop= "60_nodes_mesh_optimisation_data_storage")

    # KT_required_to_make_significant_difference(hot_bob_capacity_file="test_graph_hot_capacity",
    #                                            cold_bob_capacity_file="test_graph_cold_capacity",
    #                                            graph_edge_data_file="test_graph_edge_positions",
    #                                            graph_node_data_file="test_graph_node_positions", N=24,
    #                                            M=1, cost_det_h=1.57, cost_det_c=1.786, cost_on_h=1, cost_on_c=1.714,
    #                                            cmin=5000,
    #                                            data_storage_location_keep_each_loop=None)

    # hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
    #         hot_bob_capacity_file="test_graph_hot_capacity", cold_bob_capacity_file="test_graph_cold_capacity", cmin=5000)
    # distance_fraction = {}
    # distance_fraction_hot = {}
    # for key in hot_capacity_dict.keys():
    #     sol_dict, prob = initial_optimisation_switching(hot_key_dict = hot_capacity_dict[key], cold_key_dict = cold_capacity_dict[key], required_connections = required_connections[key], cost_det_h = 1, cost_det_c = 5, cost_on_h = 10, cost_on_c = 20, N = 50, M = 2, KT= 0,time_limit = 1e7)

    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs_nonuniform_tij(
        hot_bob_capacity_file="real_graph_hot_capacity_1",
        cold_bob_capacity_file="real_graph_cold_capacity_1", required_conn_file="real_network_key_requirements_1")

    hot_capacity_dict_no_switch, cold_capacity_dict_no_switch, required_connections_no_switch, distances = import_switched_network_values_multiple_graphs_nonuniform_tij(
        hot_bob_capacity_file="real_graph_hot_capacity_no_switch_1",
        cold_bob_capacity_file="real_graph_cold_capacity_no_switch_1", required_conn_file="real_network_key_requirements_1")
    data_storage_location_keep_each_loop = "real_graph_soln_results_with_det_values_1"
    no_soln_set = []
    differences = []
    # for cmin in np.arange(0.1, 5, 0.1):
    #     required_connections_new = {}
    #     for key in required_connections.keys():
    #         for key_2 in required_connections[key].keys():
    #             if key not in required_connections_new.keys():
    #                 required_connections_new[key] = {key_2: cmin * required_connections[key][key_2]}
    #             else:
    #                 required_connections_new[key][key_2] = cmin * required_connections[key][key_2]
    #     required_connections_new_no_switch = {}
    #     for key in required_connections_no_switch.keys():
    #         for key_2 in required_connections_no_switch[key].keys():
    #             if key not in required_connections_new_no_switch.keys():
    #                 required_connections_new_no_switch[key] = {key_2: cmin * required_connections_no_switch[key][key_2]}
    #             else:
    #                 required_connections_new_no_switch[key][key_2] = cmin * required_connections_no_switch[key][key_2]
    #
    #     for key in hot_capacity_dict.keys():
    #         if key not in no_soln_set:
    #             try:
    #                 sol_dict_2, prob_2, time_to_solve_2 = initial_optimisation_switching_fixed_switching_ratio_no_delta(
    #                     hot_key_dict=hot_capacity_dict[key],
    #                     cold_key_dict=cold_capacity_dict[key],
    #                     required_connections=required_connections_new[key],
    #                     cost_det_h=1, cost_det_c=1.136,
    #                     cost_on_h=1.68, cost_on_c=3.27, N=12, M=1,
    #                     fract_switch=0.1,
    #                     time_limit=5e3, early_stop=0.003)
    #
    #                 sol_dict, prob = optimisation_no_switching.initial_optimisation(
    #                     hot_key_dict=hot_capacity_dict_no_switch[key],
    #                     cold_key_dict=cold_capacity_dict_no_switch[key],
    #                     required_connections=required_connections_new_no_switch[key], cost_det_h=1,
    #                     cost_det_c=1.136, cost_on_h=1.68, cost_on_c=3.27, N=12, M=1)
    #
    #                 use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage  = split_sol_dict(sol_dict_2)
    #                 total_sites_used = sum(binary_dict.values())
    #                 average_detectors_per_site = sum(lambda_dict.values()) / (30)
    #                 #
    #                 if data_storage_location_keep_each_loop != None:
    #                     dictionary = [
    #                         {"cmin": cmin,"Graph key": key,"objective_value": prob_2.solution.get_objective_value(), "objective_value_no_switching": prob.solution.get_objective_value(), "average number detectors" : average_detectors_per_site}]
    #                     dictionary_fieldnames =  ["cmin", "Graph key", "objective_value", "objective_value_no_switching", "average number detectors"]
    #                     if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
    #                         with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
    #                             writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
    #                             writer.writerows(dictionary)
    #                     else:
    #                         with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
    #                             writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
    #                             writer.writeheader()
    #                             writer.writerows(dictionary)
    #
    #                 differences.append(prob_2.solution.get_objective_value() - prob.solution.get_objective_value())
    #                 print("Difference in cost of switched solution to no-switched is " + str(differences[len(differences) -1]))
    #             except:
    #                 break
    objective_values_cmin_no_switching ={}
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["cmin"] not in objective_values_cmin_no_switching.keys():
                objective_values_cmin_no_switching[row["cmin"]] = [(row["Graph key"], row["objective_value"], row["objective_value_no_switching"], row["average number detectors"])]
            else:
                objective_values_cmin_no_switching[row["cmin"]].append((row["Graph key"], row["objective_value"], row["objective_value_no_switching"], row["average number detectors"]))
    objective_values_cmin_no_switching = dict(sorted(objective_values_cmin_no_switching.items()))
    objective_values = {}
    for cmin in objective_values_cmin_no_switching.keys():
        for key, objective_value, objective_value_no_switching, average_detectors in objective_values_cmin_no_switching[cmin]:
            if cmin not in objective_values.keys():
                objective_values[cmin] = [objective_value / objective_value_no_switching]
            else:
                objective_values[cmin].append(objective_value / objective_value_no_switching)
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)


    mean_detector_vals = {}
    for cmin in objective_values_cmin_no_switching.keys():
        for key, objective_value, objective_value_no_switching, average_detectors in objective_values_cmin_no_switching[cmin]:
            if cmin not in mean_detector_vals.keys():
                mean_detector_vals[cmin] = [average_detectors]
            else:
                mean_detector_vals[cmin].append(average_detectors)
    mean_detectors = {}
    std_detectors = {}
    for key in mean_detector_vals.keys():
        mean_detectors[key] = np.mean(mean_detector_vals[key])
        std_detectors[key] = np.std(mean_detector_vals[key])
    mean_differences_det = []
    std_differences_det = []
    # topologies
    x_det = []
    for key in mean_detectors.keys():
        mean_differences_det.append(mean_detectors[key])
        std_differences_det.append(std_detectors[key])
        x_det.append(key)

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("Scale Factor Capacities", fontsize=10)
    ax1.set_ylabel("Cost of Switching Solution/Cost of No Switching Solution", fontsize=10, color = color)
    ax1.plot(x, mean_differences, color = color)

    ax2 =ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel("Total Number of Detectors / |S|(|S|-1)", fontsize = 10, color = color)
    ax2.plot(x_det, mean_differences_det, color = color)
    fig.tight_layout()
    plt.savefig("real_graph_cost_terms_with_detector_vals_from_no_averaging.png")
    plt.show()
