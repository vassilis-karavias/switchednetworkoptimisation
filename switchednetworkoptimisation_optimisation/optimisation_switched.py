import cplex
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import csv
import networkx as nx
from scipy.optimize import curve_fit
from switched_network_utils import *


########################## Add Logging Capabilities ##################################################

def log_optimal_solution_to_problem(prob, save_file, graph_id):
    sol_dict = create_sol_dict(prob)
    use_dict, lambda_dict, binary_dict = split_sol_dict(sol_dict)
    ## if file does not exist - we wish to store information of q_{i,j,d}^{m}, w_{i,j,d}^{m}, lambda_{d}^{m}, delta_{i,j,d}^{m}
    ## need to generate all possible values of i,j,d available. Need to think of the most appropriate way to store this data.
    dict = {"ID" : graph_id}
    dict.update(use_dict)
    dict.update(binary_dict)
    dict.update(lambda_dict)
    dictionary = [dict]
    dictionary_fieldnames = list(dict.keys())
    with open(save_file + '.csv', mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
        writer.writeheader()
        writer.writerows(dictionary)


########################## Post-Processing Information ################################################

def split_sol_dict(sol_dict):
    """
    Split the solution dictionary into 2 dictionaries containing the fractional usage variables only and the binary
    variables only
    Parameters
    ----------
    sol_dict : The solution dictionary containing solutions to the primary flow problem

    Returns : A dictionary with only the fractional detectors used, and a dictionary with only the binary values of
            whether the detector is on or off for cold and hot
    -------

    """
    use_dict = {}
    binary_dict = {}
    lambda_dict = {}
    for key in sol_dict:
        # get all keys that are flow and add to dictionary
        if key[0] == "q":
            use_dict[key] = sol_dict[key]
        elif key[0] == "l":
            lambda_dict[key] = sol_dict[key]
        else:
            # get the keys that are binary 'on' 'off' and add to dictionary
            binary_dict[key] = sol_dict[key]
    return use_dict, lambda_dict, binary_dict


def get_cold_binaries(binary_dict):
    """
    Returns the binaries related to on or off cooling nodes
    Parameters
    ----------
    binary_dict : The dictionary of only binary values

    Returns : A dictionary of only binary values for cooled detectors
    -------

    """
    binary_cooled_dict = {}
    for key in binary_dict:
        if key[-1] == "c":
            binary_cooled_dict[key] = binary_dict[key]
    return binary_cooled_dict

def get_hot_binaries(binary_dict):
    """
    Returns the binaries related to on or off cooling nodes
    Parameters
    ----------
    binary_dict : The dictionary of only binary values

    Returns : A dictionary of only binary values for cooled detectors
    -------

    """
    binary_cooled_dict = {}
    for key in binary_dict:
        if key[-1] == "h":
            binary_cooled_dict[key] = binary_dict[key]
    return binary_cooled_dict

def get_optimal_soln(binary_soln_pool):
    current_optimal_soln = 0
    current_optimal_value = np.infty
    for i in range(len(binary_soln_pool)):
        if binary_soln_pool[i]["objective"] < current_optimal_value:
            current_optimal_soln = i
            current_optimal_value = binary_soln_pool[i]["objective"]
    return current_optimal_soln, current_optimal_value

########################### Linear Program Constraints ##################################################




def add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N):
    """
    add the constraint: \lambda_{d}^{m} \leq N \delta_{d}^{m}
    """
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
    for detector in detectors:
        ind_flow_hot = [f"lambda_{detector}^h"]
        ind_flow_cold = [f"lambda_{detector}^c"]
        detector_hot = [f"delta_{detector}^h"]
        detector_cold = [f"delta_{detector}^c"]
        cap_const_hot = cplex.SparsePair(ind=ind_flow_hot + detector_hot, val=[1] * len(ind_flow_hot) + [-N])
        cap_const_cold = cplex.SparsePair(ind=ind_flow_cold + detector_cold, val=[1] * len(ind_flow_cold) + [-N])

        prob.linear_constraints.add(lin_expr=[cap_const_hot, cap_const_cold], senses='LL', rhs=[0,0])


def add_capacity_requirement_constraint(prob, hot_key_dict, cold_key_dict, required_connections):
    """
    adds constraint \sum_{k: d \in C} (q_{(i,j,d)}^{u}c^{u}_{(i,j,d)} + q_{(i,j,d)}^{c}c^{c}_{(i,j,d)} \geq c_{i,j}
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
    capacity_variables_hot = []
    capacity_variables_cold = []
    for key in fraction_capacity_variables:
        capacity_variables_hot.append(f"q_{key[0],key[1],key[2]}^h")
        capacity_variables_cold.append(f"q_{key[0],key[1],key[2]}^c")
    prob.variables.add(names=capacity_variables_hot, types=[prob.variables.type.continuous] * len(capacity_variables_hot), lb = [0] * len(capacity_variables_hot))
    prob.variables.add(names=capacity_variables_cold, types=[prob.variables.type.continuous] * len(capacity_variables_cold), lb = [0] * len(capacity_variables_cold))
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
        prob.linear_constraints.add(lin_expr=[constraint], senses='G', rhs=[required_connections[connection]])


def add_integer_limit(prob, hot_key_dict, cold_key_dict):
    """
    adds the constraint \sum_{i,j \in S} q_{k=(i,j,d)}^{m} \leq \lambda_{d}^{m}
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
        constraint = cplex.SparsePair(q_cold[i] + [lambda_cold[i]], val = [1] * len(q_cold[i]) + [-1])
        constraint_2 = cplex.SparsePair(q_hot[i] + [lambda_hot[i]], val = [1] * len(q_hot[i]) + [-1])
        prob.linear_constraints.add(lin_expr=[constraint, constraint_2], senses='LL', rhs = [0,0])

def add_cost_minimisation_objective(prob, hot_key_dict, cold_key_dict, cost_det_h, cost_det_c,
                                    cost_on_h, cost_on_c):
    """
    adds the cost minimisation objective \sum_{d \in C} C_{det}^{u}\lambda_{d}^{u} + C_{det}^{u}\lambda_{d}^{c} +
    \sum_{d \in C} C_{on}^{u}\delta_{d}^{u} + C_{on}^{c}\delta_{d}^{c}
    """
    obj_vals = []
    detectors = {}
    for k in hot_key_dict:
        if not detectors:
            detectors[k[2]] = 1
        elif k[2] not in detectors:
            detectors[k[2]] = 1
    # get the delta parameters and attributed cost
    for det in detectors:
        obj_vals.append((f"delta_{det}^h", cost_on_h))
        obj_vals.append((f"delta_{det}^c", cost_on_c))
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
    detectors = []
    # get the lambda parameters and attributed cost
    for key in fraction_capacity_variables:
        if key[2] not in detectors:
            obj_vals.append((f"lambda_{key[2]}^h", cost_det_h))
            obj_vals.append((f"lambda_{key[2]}^c", cost_det_c))
            detectors.append(key[2])
    # add objective function
    prob.objective.set_linear(obj_vals)
    prob.objective.set_sense(prob.objective.sense.minimize)


def add_capacity_requirement_constraint_multipath(prob, hot_key_dict, cold_key_dict, required_connections, M):
    """
    the capacity constraint for the multipath problem is similar to the non-multipath version
    \sum_{k: d \in C} (q_{(i,j,d)}^{u}c^{u}_{(i,j,d)} + q_{(i,j,d)}^{c}c^{c}_{(i,j,d)} \geq M c_{min}
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
    capacity_variables_hot = []
    capacity_variables_cold = []
    for key in fraction_capacity_variables:
        capacity_variables_hot.append(f"q_{key[0],key[1],key[2]}^h")
        capacity_variables_cold.append(f"q_{key[0],key[1],key[2]}^c")
    prob.variables.add(names=capacity_variables_hot, types=[prob.variables.type.continuous] * len(capacity_variables_hot),  lb = [0] * len(capacity_variables_hot))
    prob.variables.add(names=capacity_variables_cold, types=[prob.variables.type.continuous] * len(capacity_variables_cold),  lb = [0] * len(capacity_variables_cold))
    for connection in required_connections:
        ind_flow_hot = []
        ind_flow_cold = []
        capacities_hot = []
        capacities_cold = []
        for detector in detectors:
            ind_flow_hot.append(f"q_{connection[0],connection[1],detector}^h")
            ind_flow_cold.append(f"q_{connection[0],connection[1],detector}^c")
            capacities_hot.append(float(hot_key_dict[(connection[0], connection[1], detector)]))
            capacities_cold.append(float(cold_key_dict[(connection[0], connection[1], detector)]))
        constraint = cplex.SparsePair(ind = ind_flow_hot + ind_flow_cold, val = capacities_hot + capacities_cold)
        prob.linear_constraints.add(lin_expr=[constraint], senses='G', rhs=[M * required_connections[connection]])

def add_maximal_single_path_capacity_multipath(prob, hot_key_dict, cold_key_dict, required_connections):
    """
    adds the constraint to prevent any individual path from having a capacity greater than c_{min} - this ensures that
    at least M separate paths need to be used to meet the capacity requirements: c_{k}^{m}q_{k}^{m} \leq \frac{c_{i,j}}{M} = c_{min}
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
            capacity_hot = [float(hot_key_dict[(connection[0], connection[1], detector)])]
            capacity_cold = [float(cold_key_dict[(connection[0], connection[1], detector)])]
            if capacity_hot[0] > 0.00001 and capacity_cold[0]> 0.000001:
                constraints = [cplex.SparsePair(ind = ind_flow_hot, val = capacity_hot), cplex.SparsePair(ind = ind_flow_cold, val = capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='LL', rhs=[required_connections[connection], required_connections[connection]])
            elif capacity_cold[0] > 0.000001:
                constraints = [cplex.SparsePair(ind = ind_flow_cold, val = capacity_cold)]
                prob.linear_constraints.add(lin_expr=constraints, senses='L',
                                            rhs=[required_connections[connection]])


def create_sol_dict(prob):
    """
    Create a dictionary with the solution of the parameters
    """
    names = prob.variables.get_names()
    values = prob.solution.get_values()
    sol_dict = {names[idx]: (values[idx]) for idx in range(prob.variables.get_num())}
    return sol_dict


######################## Setup Problem ################################


def initial_optimisation(hot_key_dict, cold_key_dict, required_connections, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, time_limit = 1e7):
    """
    set up and solve the problem for mimising the number of trusted nodes

    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_capacity_requirement_constraint(prob, hot_key_dict, cold_key_dict, required_connections)
    add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N)
    add_integer_limit(prob, hot_key_dict, cold_key_dict)
    add_cost_minimisation_objective(prob, hot_key_dict, cold_key_dict, cost_det_h, cost_det_c,
                                    cost_on_h, cost_on_c)
    prob.write("test.lp")
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.dual)
    # prob.parameters.mip.limits.cutpasses.set(1)
    # prob.parameters.mip.strategy.probe.set(-1)
    # prob.parameters.mip.strategy.variableselect.set(4)
    print(prob.parameters.get_changed())
    prob.parameters.timelimit.set(time_limit)
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


def initial_optimisation_multiple_paths(hot_key_dict, cold_key_dict, required_connections, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, time_limit = 1e7, early_stop = None):
    """
    set up and solve the problem for mimising the number of trusted nodes

    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_capacity_requirement_constraint_multipath(prob, hot_key_dict, cold_key_dict, required_connections, M)
    add_integer_limit(prob, hot_key_dict, cold_key_dict)
    add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N)
    add_maximal_single_path_capacity_multipath(prob, hot_key_dict, cold_key_dict, required_connections)
    add_cost_minimisation_objective(prob, hot_key_dict, cold_key_dict, cost_det_h, cost_det_c,
                                    cost_on_h, cost_on_c)
    prob.write("test.lp")
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.dual)
    if early_stop is not None:
        prob.parameters.mip.tolerances.mipgap.set(early_stop)
    # prob.parameters.mip.limits.cutpasses.set(1)
    # prob.parameters.mip.strategy.probe.set(-1)
    # prob.parameters.mip.strategy.variableselect.set(4)
    print(prob.parameters.get_changed())
    prob.parameters.timelimit.set(time_limit)
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



def initial_optimisation_multiple_solutions(hot_key_dict, cold_key_dict, required_connections, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, time_limit = 1e7, early_stop = None):
    """
        set up and solve the problem for mimising the number of trusted nodes - this will obtain multiple solutions to
        the problem in the tolerance range desired.
    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_capacity_requirement_constraint_multipath(prob, hot_key_dict, cold_key_dict, required_connections, M)
    add_integer_limit(prob, hot_key_dict, cold_key_dict)
    add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N)
    add_maximal_single_path_capacity_multipath(prob, hot_key_dict, cold_key_dict, required_connections)
    add_cost_minimisation_objective(prob, hot_key_dict, cold_key_dict, cost_det_h, cost_det_c,
                                    cost_on_h, cost_on_c)
    prob.write("test.lp")
    prob.parameters.mip.pool.intensity = 4
    prob.parameters.mip.pool.absgap = 0.1
    prob.parameters.mip.limits.populate = 100000000
    try:
        prob.populate_solution_pool()
        numsol = prob.solution.pool.get_num()
        print(numsol)
        sol_pool = []
        for i in range(numsol):
            # get the solution dictionary for each possible solution
            soln_dict = prob.variables.get_names()
            x_i = prob.solution.pool.get_values(i)
            # objective_value = {"objective": prob.solution.pool.get_objective_value(i)}
            sol_dict = {soln_dict[idx]: (x_i[idx]) for idx in range(prob.variables.get_num())}
            sol_dict["objective"] = prob.solution.pool.get_objective_value(i)
            sol_pool.append(sol_dict)
        return sol_pool
    except:
        print("Exception raised during populate")
        return []

########################## Plots and Investigations ##############################


def check_infeasible(prob):
    """Checks solved cplex problem for infeasibility. Returns True when infeasible, otherwise false."""
    status = prob.solution.get_status()
    if status == 3:
        return True
    else:
        return False


def check_timeout(prob):
    """Checks solved cplex problem for timeout. Returns True when timed out, otherwise false."""
    status = prob.solution.get_status()
    if status == 11:
        return True
    else:
        return False


def check_solvable(prob):
    """Checks solved cplex problem for timeout, infeasibility or optimal solution. Returns True when feasible solution obtained."""
    status = prob.solution.get_status()
    if status == 101 or status == 102: # proven optimal or optimal within tolerance
        return True
    elif status == 103:  # proven infeasible or Timeout
        return False
    elif status == 107:  # timed out
        print("Optimiser Timed out - assuming infeasible")
        return True
    else:
        print(f"Unknown Solution Status: {status} - assuming infeasible")
        return False


def plot_graph(graph, binary_dict):
    graph = graph.to_undirected()
    pos = {}
    for node in graph.nodes:
        pos[node] = [graph.nodes[node]["xcoord"], graph.nodes[node]["ycoord"]]
    plt.figure()
    detector_nodes_list = []
    source_node_list = []
    for node in graph.nodes:
        if graph.nodes[node]["type"] == "B":
            detector_nodes_list.append(node)
        else:
            source_node_list.append(node)
    nx.draw_networkx_nodes(graph, pos, nodelist=source_node_list, node_color="k")
    all_nodes = []
    on_nodes_hot = []
    on_nodes_cold = []
    #### consider this but for cold and hot.....
    for key in binary_dict:
        current_node = int(key[6:-2])
        on_off = int(binary_dict[key])
        temp = key[-1]
        all_nodes.append(current_node)
        if on_off == 1 and temp == "h":
            on_nodes_hot.append(current_node)
        elif on_off == 1 and temp == "c":
            on_nodes_cold.append(current_node)
    on_nodes_both = list(set(on_nodes_hot) & set(on_nodes_hot))
    on_nodes_hot_only = list(set(on_nodes_hot) - set(on_nodes_both))
    on_nodes_cold_only = list(set(on_nodes_cold) - set(on_nodes_both))
    off_nodes =  list(set(all_nodes) - set(on_nodes_both) - set(on_nodes_cold_only) - set(on_nodes_hot_only))
    nx.draw_networkx_nodes(graph, pos, nodelist=on_nodes_both, node_shape="d", label = "Both Hot and Cold on")
    nx.draw_networkx_nodes(graph, pos, nodelist=on_nodes_hot_only, node_shape="v", label = "Only Hot on")
    nx.draw_networkx_nodes(graph, pos, nodelist=on_nodes_cold_only, node_shape="^", label = "Only Cold on")
    nx.draw_networkx_nodes(graph, pos, nodelist=off_nodes, node_shape="o", label = "Node Off")
    nx.draw_networkx_edges(graph, pos, edge_color="k")
    plt.axis("off")
    plt.legend(loc = "best", fontsize = "small")
    plt.show()


def get_number_of_nodes(hot_capacity_dict, cold_capacity_dict):
    number_nodes = 0
    for key in hot_capacity_dict.keys():
        if key[0] > number_nodes:
            number_nodes = key[0]
        if key[1] > number_nodes:
            number_nodes = key[1]
        if key[2] > number_nodes:
            number_nodes = key[2]
    for key in cold_capacity_dict.keys():
        if key[0] > number_nodes:
            number_nodes = key[0]
        if key[1] > number_nodes:
            number_nodes = key[1]
        if key[2] > number_nodes:
            number_nodes = key[2]
    return number_nodes # not measured from 0.


def get_cooling_effective_point(hot_bob_capacity_file, cold_bob_capacity_file,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, cmin = 10000):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    distance_fraction = {}
    distance_fraction_hot = {}
    for key in hot_capacity_dict.keys():
        try:
            sol_dict, prob = initial_optimisation_multiple_paths(hot_key_dict = hot_capacity_dict[key], cold_key_dict = cold_capacity_dict[key], required_connections = required_connections[key], cost_det_h = cost_det_h, cost_det_c = cost_det_c, cost_on_h = cost_on_h, cost_on_c = cost_on_c, N = N, M = M, time_limit = 1e7)
            if check_solvable(prob):
                use_dict, lambda_dict, binary_dict = split_sol_dict(sol_dict)
                distance = distances[key]
                cooled_dict = get_cold_binaries(binary_dict)
                hot_dict = get_hot_binaries(binary_dict)
                i = 0
                j = 0
                for key1 in cooled_dict:
                    if cooled_dict[key1] > 0.5:
                        i += 1
                for key1 in hot_dict:
                     if hot_dict[key1] > 0.5:
                        j += 1
                fraction_of_on = i/len(cooled_dict)
                fraction_of_on_hot = j/len(hot_dict)
                if distance in distance_fraction.keys():
                    distance_fraction[distance].append(fraction_of_on)
                else:
                    distance_fraction[distance] = [fraction_of_on]
                if distance in distance_fraction_hot.keys():
                    distance_fraction_hot[distance].append(fraction_of_on_hot)
                else:
                    distance_fraction_hot[distance] = [fraction_of_on_hot]
        except:
            continue
    # at what point does cooling become economically viable? - when the cheapest solution involves having cooling >50%
    # of the time
    for distance in distance_fraction:
        cooled_total = sum(distance_fraction[distance])
        hot_total = sum(distance_fraction_hot[distance])
        if cooled_total > hot_total:
            return distance

def cost_of_network_increase_number_nodes(hot_bob_capacity_file, cold_bob_capacity_file,   graph_edge_data_file, graph_node_data_file, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, cmin = 1000):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    cost_network = {}
    for key in hot_capacity_dict.keys():
        try:
            sol_dict, prob = initial_optimisation_multiple_paths(hot_key_dict=hot_capacity_dict[key],
                                                                 cold_key_dict=cold_capacity_dict[key],
                                                                 required_connections=required_connections[key],
                                                                 cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                 cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                 time_limit=1e7)
            if check_solvable(prob):
                use_dict, lambda_dict, binary_dict = split_sol_dict(sol_dict)
                number_nodes = get_number_of_nodes(hot_capacity_dict[key], cold_capacity_dict[key])
                plot_graph(graphs[key], binary_dict)
                if number_nodes in cost_network.keys():
                    cost_network[number_nodes].append(prob.solution.get_objective_value())
                else:
                    cost_network[number_nodes] = [prob.solution.get_objective_value()]
        except:
            continue
    network_costs_mean_std = {}
    x = []
    y = []
    yerr = []
    for key in cost_network:
        network_costs_mean_std[key] = [np.mean(cost_network[key]), np.std(cost_network[key])]
        x.append(key)
        y.append(network_costs_mean_std[key][0])
        yerr.append(network_costs_mean_std[key][1])
    plt.errorbar(x, y, yerr=yerr)
    plt.xlabel("Number of Nodes in Network", fontsize=10)
    plt.ylabel("Cost of Network", fontsize=10)
    plt.savefig("cost_of_network_with_increasing_number_of_nodes.png")
    plt.show()


def cost_of_network_increase_distance(hot_bob_capacity_file, cold_bob_capacity_file, cost_det_h, cost_det_c, cost_on_h,
                                      cost_on_c, N, M, cmin=5000):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)

    cost_network = {}
    for key in hot_capacity_dict.keys():
        try:
            sol_dict, prob = initial_optimisation_multiple_paths(hot_key_dict=hot_capacity_dict[key],
                                                                 cold_key_dict=cold_capacity_dict[key],
                                                                 required_connections=required_connections[key],
                                                                 cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                 cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                 time_limit=1e7)
            if check_solvable(prob):
                use_dict, lambda_dict, binary_dict = split_sol_dict(sol_dict)
                distance = distances[key]
                if distance in cost_network.keys():
                    cost_network[distance].append(prob.solution.get_objective_value())
                else:
                    cost_network[distance] = [prob.solution.get_objective_value()]
        except:
            continue
    network_costs_mean_std = {}
    x = []
    y = []
    yerr = []
    for key in cost_network:
        network_costs_mean_std[key] = [np.mean(cost_network[key]), np.std(cost_network[key])]
        x.append(key)
        y.append(network_costs_mean_std[key][0])
        yerr.append(network_costs_mean_std[key][1])
    plt.errorbar(x, y, yerr=yerr)
    plt.xlabel("Distance of Network/Km", fontsize=10)
    plt.ylabel("Cost of Network", fontsize=10)
    plt.savefig("cost_of_network_with_increasing_distance.png")
    plt.show()


def exponential_fit(x, a, exp, c):
    return a * np.exp(exp * x) + c

def polynomial(x, a, b):
    return a * (x ** 3) + b


def time_taken_with_increasing_number_of_nodes(hot_bob_capacity_file, cold_bob_capacity_file,  graph_edge_data_file, graph_node_data_file, cost_det_h, cost_det_c, cost_on_h,
                                      cost_on_c, N, M, cmin=5000):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)

    time_taken = {}
    for key in hot_capacity_dict.keys():
        try:
            t_0 = time.time()
            sol_dict, prob = initial_optimisation_multiple_paths(hot_key_dict=hot_capacity_dict[key],
                                                                     cold_key_dict=cold_capacity_dict[key],
                                                                     required_connections=required_connections[key],
                                                                     cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                     cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                     time_limit=1e7)
            if check_solvable(prob):
                use_dict, lambda_dict, binary_dict = split_sol_dict(sol_dict)
                t_1 = time.time()
                number_nodes = get_number_of_nodes(hot_capacity_dict[key], cold_capacity_dict[key])
                plot_graph(graphs[key], binary_dict)
            if number_nodes in time_taken.keys():
                time_taken[number_nodes].append(t_1 - t_0)
            else:
                time_taken[number_nodes] = [t_1 - t_0]
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
    x_exponential = x[:int(np.ceil(len(x)/3))]
    y_exponential = y[:int(np.ceil(len(x)/3))]
    popt, pcov = curve_fit(exponential_fit, x_exponential, y_exponential)
    x_exponential = np.arange(x[0], x[-1], 0.1)
    y_exponential = [exponential_fit(a, popt[0], popt[1]) for a in x_exponential]
    # fit of polynomial
    popt_poly, pcov_poly = curve_fit(polynomial, x, y)
    y_poly = [polynomial(a, popt_poly[0], popt_poly[1]) for a in x_exponential]

    plt.errorbar(x, y, yerr=yerr, color="r")
    plt.plot(x_exponential[:int(np.ceil(len(x_exponential)/1.25))], y_exponential[:int(np.ceil(len(x_exponential)/1.25))], color = "b")
    plt.plot(x_exponential, y_poly, color = "k")
    plt.xlabel("No. Nodes in Graph", fontsize=10)
    plt.ylabel("Time/s", fontsize=10)
    plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("time_investigation_bus_topology.png")
    plt.show()



def get_region_solution(hot_bob_capacity_file, cold_bob_capacity_file, cost_det_h, cost_det_c, cost_on_h,
                                      cost_on_c, N, M, cmin=5000):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)

    for key in hot_capacity_dict.keys():
        try:
            t_0 = time.time()
            soln_pool = initial_optimisation_multiple_solutions(hot_key_dict=hot_capacity_dict[key],
                                                                 cold_key_dict=cold_capacity_dict[key],
                                                                 required_connections=required_connections[key],
                                                                 cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                 cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                 time_limit=1e7)
            binary_soln_pool = []
            for soln in soln_pool:
                q_solns, lambda_soln, binary_solns = split_sol_dict(soln)
                binary_soln_pool.append(binary_solns)
            optimal_soln_position, optimal_soln_value = get_optimal_soln(binary_soln_pool)
            optimal_soln = binary_soln_pool[optimal_soln_position]

        except:
            print("Error in Optimisation Program")


def get_early_stop_difference(hot_bob_capacity_file, cold_bob_capacity_file, graph_edge_data_file, graph_node_data_file, cost_det_h, cost_det_c, cost_on_h,
                                      cost_on_c, N, M, cmin=5000, time_limit_early_stop = 1e2, early_stop = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    graphs = import_graph_structure(node_information = graph_node_data_file, edge_information = graph_edge_data_file)
    difference_early_stop_array = []
    for key in hot_capacity_dict.keys():
        try:
            t_0 = time.time()
            sol_dict, prob = initial_optimisation_multiple_paths(hot_key_dict=hot_capacity_dict[key],
                                                                 cold_key_dict=cold_capacity_dict[key],
                                                                 required_connections=required_connections[key],
                                                                 cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                 cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                 time_limit=1e7)
            sol_dict_early_stop, prob_early_stop = initial_optimisation_multiple_paths(hot_key_dict=hot_capacity_dict[key],
                                                                 cold_key_dict=cold_capacity_dict[key],
                                                                 required_connections=required_connections[key],
                                                                 cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                 cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                 time_limit=time_limit_early_stop, early_stop = early_stop)

            difference_early_stop = (prob_early_stop.solution.get_objective_value() - prob.solution.get_objective_value())/prob.solution.get_objective_value()
            difference_early_stop_array.append(difference_early_stop)
            q_solns, lambda_solns, binary_solns = split_sol_dict(sol_dict)
            graph = graphs[key]
            # plot_graph(graph, binary_solns)
            q_solns, lambda_solns, binary_solns_early = split_sol_dict(sol_dict_early_stop)
            # plot_graph(graph, binary_solns_early)
        except:
            print("Error in the optimisation")
    print(difference_early_stop_array)
    print("mean: " + str(np.mean(difference_early_stop_array)))
    print("std: " + str(np.std(difference_early_stop_array)))


def find_critical_ratio_cooled_uncooled(hot_bob_capacity_file, cold_bob_capacity_file, graph_edge_data_file, graph_node_data_file, N, M, ratio_node_detector, cmin=5000, data_storage_location_keep_each_loop = None):
    # things that affect critical ratio: number of nodes and connectivity of graph

    cost_on_h = 1.0
    cost_det_h = cost_on_h * ratio_node_detector
    ratio_hot_cold = 1
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file, cold_bob_capacity_file=cold_bob_capacity_file, cmin=cmin)
    graphs = import_graph_structure(node_information = graph_node_data_file, edge_information = graph_edge_data_file)
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
            last_ratio_done = 0
            number_of_rows = 0
            dictionary_fieldnames = ["ratio_hot_cold","number_nodes","cold_nodes_on","hot_nodes_on"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        last_ratio_done = 0
        number_of_rows = 0
    for ratio_hot_cold in np.arange(last_ratio_done,5,0.2):
        cost_on_c = cost_on_h * ratio_hot_cold
        cost_det_c = cost_det_h * ratio_hot_cold
        cold_node_on_number_nodes = {}
        hot_node_on_number_nodes = {}
        for key in hot_capacity_dict.keys():
            if number_of_rows != 0:
                number_of_rows -= 1
                continue
            if key not in no_soln_set:
                try:
                    t_0 = time.time()
                    sol_dict, prob = initial_optimisation_multiple_paths(hot_key_dict=hot_capacity_dict[key],
                                                                         cold_key_dict=cold_capacity_dict[key],
                                                                         required_connections=required_connections[key],
                                                                         cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                                                                         cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                                                                         time_limit=5e2, early_stop=0.003)
                    if check_solvable(prob):
                        q_solns, lambda_soln, binary_solns = split_sol_dict(sol_dict)
                        number_nodes = get_number_of_nodes(hot_capacity_dict[key], cold_capacity_dict[key])
                        cold_binaries = list(get_cold_binaries(binary_solns).values())
                        hot_binaries = list(get_hot_binaries(binary_solns).values())
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
        cmap[key] = np.array([hot_values, 0, cold_values, 1])
    x = []
    y = []
    cmap_array = []
    for key in cmap.keys():
        x.append(key[0])
        y.append(key[1])
        cmap_array.append(cmap[key])

    plt.scatter(x, y, c=cmap_array)
    plt.xlabel("Ratio of cost of Uncooled Detectors to Cooled Detectors")
    plt.ylabel("Number of Nodes in the Graph")
    plt.savefig("heat_map_mesh_topology_35.png")
    plt.show()


# get_early_stop_difference(hot_bob_capacity_file="bus_topology_hot_capacity_data", cold_bob_capacity_file="bus_topology_cold_capacity_data", graph_edge_data_file = "bus_topology_edge_position_data", graph_node_data_file = "bus_topology_node_position_data", cost_det_c= 0.2, cost_det_h=0.1, cost_on_c=10, cost_on_h=1, N= 10, M = 2, cmin=5000, early_stop=0.1)
# find_critical_ratio_cooled_uncooled(hot_bob_capacity_file="capacity_hot_20to40_5connections", cold_bob_capacity_file="capacity_cold_20to40_5connections", graph_edge_data_file = "graphs_edge_data_20to40_5connections", graph_node_data_file = "graphs_node_data_20to40_5connections", N= 10, M = 2, ratio_node_detector = 0.1, cmin=5000)
# time_taken_with_increasing_number_of_nodes(hot_bob_capacity_file="bus_topology_hot_capacity_data", cold_bob_capacity_file="bus_topology_cold_capacity_data", graph_edge_data_file = "bus_topology_edge_position_data", graph_node_data_file = "bus_topology_node_position_data", cost_det_h = 0.2, cost_det_c = 0.2, cost_on_h = 1, cost_on_c = 5, N = 50, M = 2)


if  __name__ == "__main__":
    find_critical_ratio_cooled_uncooled(hot_bob_capacity_file="mesh_topology_35_hot_capacity_data_complete_final", cold_bob_capacity_file="mesh_topology_35_cold_capacity_data_complete_final", graph_edge_data_file = "mesh_topology_35_edge_position_data_complete_final", graph_node_data_file = "mesh_topology_35_node_position_data_complete_final", N = 50, M = 2, ratio_node_detector=0.1, data_storage_location_keep_each_loop = "mesh_optimisation_data_storage")




# different_no_nodes
# get_early_stop_difference(hot_bob_capacity_file="capacity_hot_20", cold_bob_capacity_file="capacity_cold_20", graph_edge_data_file = "graphs_edge_data_20", graph_node_data_file = "graphs_node_data_20", cost_det_h = 0.2, cost_det_c = 0.2, cost_on_h = 1, cost_on_c = 2, N = 10, M = 2,
#                           early_stop = 0.10)

# get_region_solution(hot_bob_capacity_file="capacity_hot_12_different_sizes", cold_bob_capacity_file="capacity_cold_12_different_sizes", cost_det_h = 0.2, cost_det_c = 0.2, cost_on_h = 1, cost_on_c = 2, N = 10, M = 2)
# P(h1ot_key_dict = hot_capacity_dict[key], cold_key_dict = cold_capacity_dict[key], required_connections = required_connections[key], cost_det_h = 0.5, cost_det_c = 0.1, cost_on_h = 1, cost_on_c = 10, N = 100, M = 2, time_limit = 1e7)