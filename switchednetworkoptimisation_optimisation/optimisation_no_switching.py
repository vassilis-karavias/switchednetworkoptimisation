import cplex
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import csv
import networkx as nx
from scipy.optimize import curve_fit
from switched_network_utils import *

def create_sol_dict(prob):
    """
    Create a dictionary with the solution of the parameters
    """
    names = prob.variables.get_names()
    values = prob.solution.get_values()
    sol_dict = {names[idx]: (values[idx]) for idx in range(prob.variables.get_num())}
    return sol_dict


def add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N):
    binary_node_variables = []
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    for det in detectors:
        binary_node_variables.append(f"delta_{det}^h")
        binary_node_variables.append(f"delta_{det}^c")
    prob.variables.add(names=binary_node_variables, types=[prob.variables.type.binary] * len(binary_node_variables))

    fraction_capacity_variables = {}
    for k in hot_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[(k, "h")] = 1
        elif (k,"h") not in fraction_capacity_variables:
            fraction_capacity_variables[(k, "h")] = 1
    for k in cold_key_dict:
        if not fraction_capacity_variables:
            fraction_capacity_variables[(k, "c")] = 1
        elif (k,"c") not in fraction_capacity_variables:
            fraction_capacity_variables[(k, "c")] = 1

    capacity_integers = []
    capacity_continuous = []
    for key, temp in fraction_capacity_variables:
        capacity_integers.append(f"lambda_{key[0],key[1],key[2]}^{temp}")
        capacity_continuous.append(f"q_{key[0],key[1],key[2]}^{temp}")
    prob.variables.add(names=capacity_integers,
                       types=[prob.variables.type.integer] * len(capacity_integers))
    prob.variables.add(names=capacity_continuous,
                       types=[prob.variables.type.continuous] * len(capacity_integers))
    temp = ["h", "c"]
    for det in detectors.keys():
        for t in temp:
            lambda_variables = []
            delta_var = [f"delta_{det}^{t}"]
            for source, target in required_connections.keys():
                lambda_variables.append(f"lambda_{source, target, det}^{t}")
            constraint = cplex.SparsePair(ind=lambda_variables + delta_var, val=[1]* len(lambda_variables) + [-N])
            prob.linear_constraints.add(lin_expr=[constraint], senses='L', rhs=[0.0])

def add_capacity_requirement_constraint(prob, hot_key_dict, cold_key_dict, required_connections, M):
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    for connection in required_connections:
        ind_flow_hot = []
        ind_flow_cold = []
        capacities_hot = []
        capacities_cold = []
        for detector in detectors:
            ind_flow_hot.append(f"q_{connection[0],connection[1],detector}^h")
            ind_flow_cold.append(f"q_{connection[0],connection[1],detector}^c")
            capacities_hot.append(hot_key_dict[(connection[0], connection[1], detector)])
            capacities_cold.append(cold_key_dict[(connection[0], connection[1], detector)])
        constraint = cplex.SparsePair(ind = ind_flow_hot + ind_flow_cold, val = capacities_hot + capacities_cold)
        prob.linear_constraints.add(lin_expr=[constraint], senses='G', rhs=[M * required_connections[connection]])

def add_capacity_requirement_constraint_multipath(prob, hot_key_dict, cold_key_dict, required_connections):
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    for connection in required_connections:
        for detector in detectors:
            ind_flow_hot = [f"q_{connection[0], connection[1], detector}^h"]
            ind_flow_cold = [f"q_{connection[0], connection[1], detector}^c"]
            capacity_hot = [float(hot_key_dict[(connection[0], connection[1], detector)])]
            capacity_cold = [float(cold_key_dict[(connection[0], connection[1], detector)])]
            if capacity_hot[0] > 0.000001 and capacity_cold[0] > 0.000001:
                constraints = [cplex.SparsePair(ind=ind_flow_hot, val=capacity_hot),
                               cplex.SparsePair(ind=ind_flow_cold, val=capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='LL',
                                            rhs=[required_connections[connection], required_connections[connection]])
            elif capacity_cold[0] > 0.000001:
                constraints = [cplex.SparsePair(ind=ind_flow_cold, val=capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='L',
                                            rhs=[required_connections[connection]])

def add_continuous_discrete_limit(prob, hot_key_dict, cold_key_dict):
    for key in hot_key_dict.keys():
        vals = [f"lambda_{key[0],key[1],key[2]}^h", f"q_{key[0],key[1],key[2]}^h"]
        coeffs = [1,-1]
        constraints = [cplex.SparsePair(ind=vals, val=coeffs)]
        prob.linear_constraints.add(lin_expr=constraints, senses='G',
                                        rhs=[0.0])
    for key in cold_key_dict.keys():
        vals = [f"lambda_{key[0],key[1],key[2]}^c", f"q_{key[0],key[1],key[2]}^c"]
        coeffs = [1,-1]
        constraints = [cplex.SparsePair(ind=vals, val=coeffs)]
        prob.linear_constraints.add(lin_expr=constraints, senses='G',
                                        rhs=[0.0])

def add_objective_value(prob, hot_key_dict, cold_key_dict, cost_det_h, cost_det_c, cost_on_h, cost_on_c):
    obj_vals = []
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    for key in hot_key_dict.keys():
        obj_vals.append((f"lambda_{key[0],key[1],key[2]}^h", cost_det_h))
    for key in cold_key_dict.keys():
        obj_vals.append((f"lambda_{key[0],key[1],key[2]}^c", cost_det_c))
    for det in detectors.keys():
        obj_vals.append((f"delta_{det}^h", cost_on_h))
        obj_vals.append((f"delta_{det}^c", cost_on_c))
    prob.objective.set_linear(obj_vals)
    prob.objective.set_sense(prob.objective.sense.minimize)



def initial_optimisation(hot_key_dict, cold_key_dict, required_connections, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, time_limit = 1e7, early_stop = 0.003):
    """
    set up and solve the problem for mimising the number of trusted nodes

    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N)
    add_capacity_requirement_constraint(prob, hot_key_dict, cold_key_dict, required_connections, M)
    add_capacity_requirement_constraint_multipath(prob, hot_key_dict, cold_key_dict, required_connections)
    add_continuous_discrete_limit(prob, hot_key_dict, cold_key_dict)
    add_objective_value(prob, hot_key_dict, cold_key_dict, cost_det_h, cost_det_c, cost_on_h, cost_on_c)
    prob.write("test.lp")
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.dual)
    # prob.parameters.mip.limits.cutpasses.set(1)
    # prob.parameters.mip.strategy.probe.set(-1)
    # prob.parameters.mip.strategy.variableselect.set(4)
    print(prob.parameters.get_changed())
    prob.parameters.timelimit.set(time_limit)
    if early_stop is not None:
        prob.parameters.mip.tolerances.mipgap.set(early_stop)
    t_1 = time.time()
    print("Time to set up problem: " + str(t_1-t_0))
    prob.solve()
    t_2 = time.time()
    print("Time to solve problem: " + str(t_2 - t_1))
    print(f"The Minimum Number of Trusted Nodes: {prob.solution.get_objective_value()}")
    print(f"Number of Variables = {prob.variables.get_num()}")
    print(f"Number of Conditions = {prob.linear_constraints.get_num()}")
    sol_dict = create_sol_dict(prob)

    return sol_dict, prob


if  __name__ == "__main__":
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file="8_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file="8_nodes_mesh_topology_35_cold_capacity", cmin=1000)
    sol_dict, prob = initial_optimisation(hot_key_dict=hot_capacity_dict[1], cold_key_dict=cold_capacity_dict[1], required_connections=required_connections[1], cost_det_h = 1, cost_det_c = 1.172, cost_on_h = 1.8, cost_on_c = 3.27, N = 2400, M = 2)