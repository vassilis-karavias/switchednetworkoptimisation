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
import optimisation_no_switching
import Relaxed_heuristic
import LP_relaxation
import optimisation_switching_model

def add_integer_limit_fixed_number_switch(prob, hot_key_dict, cold_key_dict, fract_switch, omega):
    """
    adds the constraint \sum_{i,j \in S} \frac{\Omega_{k=(i,j,d)}^{m}}{\omega} \leq (1-fract_switch) *\lambda_{d}^{m} - in the program we use w to mean q'
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
    omega_terms_hot = []
    omega_terms_cold = []
    for key in fraction_capacity_variables:
        if key[2] not in detectors:
            capacity_integers_hot.append(f"lambda_{key[2]}^h")
            capacity_integers_cold.append(f"lambda_{key[2]}^c")
            detectors.append(key[2])
        omega_terms_cold.append(f"Omega_{key[0],key[1],key[2]}^c")
        omega_terms_hot.append(f"Omega_{key[0], key[1], key[2]}^h")
    prob.variables.add(names=capacity_integers_hot,
                       types=[prob.variables.type.integer] * len(capacity_integers_hot))
    prob.variables.add(names=capacity_integers_cold,
                       types=[prob.variables.type.integer] * len(capacity_integers_cold))
    prob.variables.add(names=omega_terms_cold,
                       types=[prob.variables.type.integer] * len(omega_terms_cold))
    prob.variables.add(names=omega_terms_hot,
                       types=[prob.variables.type.integer] * len(omega_terms_hot))

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
            q_cold.append([f"Omega_{key[0],key[1],key[2]}^c"])
            q_hot.append([f"Omega_{key[0],key[1],key[2]}^h"])
        else:
            for i in range(len(new_detectors)):
                if key[2] == new_detectors[i]:
                    q_cold[i].append(f"Omega_{key[0],key[1],key[2]}^c")
                    q_hot[i].append(f"Omega_{key[0],key[1],key[2]}^h")
    for i in range(len(q_cold)):
        constraint = cplex.SparsePair(q_cold[i] + [lambda_cold[i]], val = [1/omega] * len(q_cold[i]) + [-(1-fract_switch)])
        constraint_2 = cplex.SparsePair(q_hot[i] + [lambda_hot[i]], val = [1/omega] * len(q_hot[i]) + [-(1-fract_switch)])
        prob.linear_constraints.add(lin_expr=[constraint, constraint_2], senses='LL', rhs = [0,0])



def add_q_omega_relationship(prob, hot_key_dict, cold_key_dict, omega):
    for key in hot_key_dict.keys():
        q_value = [f"q_{key[0], key[1], key[2]}^h"]
        omega_value = [f"Omega_{key[0], key[1], key[2]}^h"]
        constraint = cplex.SparsePair(q_value + omega_value, val = [1,-1/omega])
        prob.linear_constraints.add(lin_expr = [constraint], senses = "L", rhs = [0])
    for key in cold_key_dict.keys():
        q_value = [f"q_{key[0], key[1], key[2]}^c"]
        omega_value = [f"Omega_{key[0], key[1], key[2]}^c"]
        constraint = cplex.SparsePair(q_value + omega_value, val = [1,-1/omega])
        prob.linear_constraints.add(lin_expr = [constraint], senses = "L", rhs = [0])


def initial_optimisation_switching_limited_number_of_switching(hot_key_dict, cold_key_dict, required_connections, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M,omega, fract_switch = 0.9,  time_limit = 1e7, early_stop = None):
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    optimisation_switched.add_capacity_requirement_constraint_multipath(prob, hot_key_dict, cold_key_dict, required_connections, M)
    add_integer_limit_fixed_number_switch(prob, hot_key_dict, cold_key_dict, fract_switch, omega)
    optimisation_switched.add_on_node_constraint(prob, hot_key_dict, cold_key_dict, required_connections, N)
    add_q_omega_relationship(prob, hot_key_dict, cold_key_dict, omega)
    optimisation_switching_model.add_maximal_single_path_capacity_multipath_no_w_term_no_delta_term(prob, hot_key_dict, cold_key_dict, required_connections)
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



def plot_solution_for_varying_capacity_with_omega(hot_bob_capacity_file, cold_bob_capacity_file, hot_bob_capacity_file_no_switch, cold_bob_capacity_file_no_switch, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, f_switch, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_no_switch = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file,
        cold_bob_capacity_file=cold_bob_capacity_file, cmin=1000)

    hot_capacity_dict_no_switch, cold_capacity_dict_no_switch, required_connections_no_switch, distances_no_switch = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file_no_switch,
        cold_bob_capacity_file=cold_bob_capacity_file_no_switch, cmin=1000)
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
    objective_values_cmin = {}
    lambda_values = {}
    for cmin in np.arange(start=500, stop=5000, step=500):
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
                    continue
                try:
                    for req in required_connections[key].keys():
                        required_connections[key][req] = float(cmin)
                    sol_dict, prob, time_to_solve = optimisation_switching_model.initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=f_switch,
                        time_limit=1e3, early_stop=0.003)
                except:
                    no_soln_set.append(key)
                    continue
                if optimisation_switched.check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if cmin not in objective_values_cmin.keys():
                        objective_values_cmin[cmin] = {key: objective_value}
                    else:
                        objective_values_cmin[cmin][key] = objective_value
                    # print results for on nodes:
                    use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = optimisation_switching_model.split_sol_dict(sol_dict)
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

    if data_storage_location_keep_each_loop_no_switch != None:
        if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["cmin"]
            dataframe_of_cmin_done = plot_information[plot_information["cmin"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            cmin_current = last_ratio_done.iloc[0]
        else:
            cmin_current = None
            current_key = None
            dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        cmin_current = None
        current_key = None
    no_soln_set = []
    objective_values_no_switch = {}
    lambda_values = {}
    for cmin in np.arange(start=500, stop=5000, step=500):
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
                    continue
                try:
                    for req in required_connections[key].keys():
                        required_connections[key][req] = float(cmin)
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
                    if cmin not in objective_values_no_switch.keys():
                        objective_values_no_switch[cmin] = {key: objective_value}
                    else:
                        objective_values_no_switch[cmin][key] = objective_value
                    # print results for on nodes:
                    use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = optimisation_switching_model.split_sol_dict(sol_dict)
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
                    if data_storage_location_keep_each_loop_no_switch != None:
                        dictionary = [
                            {"cmin": cmin, "Graph key": key, "objective_value": objective_value}]
                        dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
                            with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)




    omega =[2,3,4,5]
    objective_values_omega = {}
    for o in omega:
        if data_storage_location_keep_each_loop != None:
            if os.path.isfile(data_storage_location_keep_each_loop + str(o) + '.csv'):
                plot_information = pd.read_csv(data_storage_location_keep_each_loop + str(o) + ".csv")
                last_row_explored = plot_information.iloc[[-1]]
                last_ratio_done = last_row_explored["cmin"]
                dataframe_of_cmin_done = plot_information[plot_information["cmin"] == last_ratio_done.iloc[0]]
                current_key = last_row_explored["Graph key"].iloc[0]
                cmin_current = last_ratio_done.iloc[0]
            else:
                cmin_current = None
                current_key = None
                dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
                with open(data_storage_location_keep_each_loop + str(o) + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                    writer.writeheader()
        else:
            cmin_current = None
            current_key = None
        no_soln_set = []
        objective_values_cmin = {}
        lambda_values = {}
        for cmin in np.arange(start=500, stop=5000, step=500):
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
                        continue
                    try:
                        for req in required_connections[key].keys():
                            required_connections[key][req] = float(cmin)
                        sol_dict, prob, time_to_solve = initial_optimisation_switching_limited_number_of_switching(hot_key_dict=hot_capacity_dict[key],
                            cold_key_dict=cold_capacity_dict[key],
                            required_connections=required_connections[key],
                            cost_det_h=cost_det_h, cost_det_c=cost_det_c,
                            cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                            omega = o,
                            fract_switch=f_switch,
                            time_limit=1e3, early_stop=0.003)
                    except:
                        no_soln_set.append(key)
                        continue
                    if optimisation_switched.check_solvable(prob):
                        objective_value = prob.solution.get_objective_value()
                        if o not in objective_values_omega.keys():
                            objective_values_omega[o] = {cmin: {key: objective_value}}
                        elif cmin not in objective_values_omega[o].keys():
                            objective_values_omega[o][cmin] = {key: objective_value}
                        else:
                            objective_values_omega[o][cmin][key] = objective_value
                        # print results for on nodes:
                        use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = optimisation_switching_model.split_sol_dict(
                            sol_dict)
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
                            if os.path.isfile(data_storage_location_keep_each_loop + str(o) + '.csv'):
                                with open(data_storage_location_keep_each_loop +str(o) + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writerows(dictionary)
                            else:
                                with open(data_storage_location_keep_each_loop + str(o) + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writeheader()
                                    writer.writerows(dictionary)


    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["cmin"] not in objective_values_cmin.keys():
                objective_values_cmin[row["cmin"]] = {row["Graph key"] : row["objective_value"]}
            else:
                objective_values_cmin[row["cmin"]][row["Graph key"]] = row["objective_value"]
        for o in omega:
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + str(o) + ".csv")
            for index, row in plot_information.iterrows():
                if o not in objective_values_omega.keys():
                    objective_values_omega[o] = {row["cmin"]: {row["Graph key"] : row["objective_value"]}}
                elif row["cmin"] not in objective_values_omega[o].keys():
                    objective_values_omega[o][row["cmin"]] = {row["Graph key"]: row["objective_value"]}
                else:
                    objective_values_omega[o][row["cmin"]][row["Graph key"]]= row["objective_value"]
    if data_storage_location_keep_each_loop_no_switch != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
        for index, row in plot_information.iterrows():
            if row["cmin"] not in objective_values_no_switch.keys():
                objective_values_no_switch[row["cmin"]] = {row["Graph key"]: row["objective_value"]}
            else:
                objective_values_no_switch[row["cmin"]][row["Graph key"]] = row["objective_value"]
    objective_values_cmin_ratios = {}
    for cmin in objective_values_cmin.keys():
        for key in objective_values_cmin[cmin].keys():
            if cmin not in objective_values_cmin_ratios.keys():
                if objective_values_no_switch[cmin][key] > 0.0000001:
                    objective_values_cmin_ratios[cmin] = [objective_values_cmin[cmin][key]/objective_values_no_switch[cmin][key]]
            else:
                if objective_values_no_switch[cmin][key] > 0.0000001:
                    objective_values_cmin_ratios[cmin].extend([objective_values_cmin[cmin][key] / objective_values_no_switch[cmin][key]])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values_cmin_ratios.keys():
        mean_objectives[key] = np.mean(objective_values_cmin_ratios[key])
        std_objectives[key] = np.std(objective_values_cmin_ratios[key])
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
        # cross_val_scores[fun]= cross_val_score(y_values = mean_differences, x_values = x, k = 5, model = getsource(fun).split(" ")[1].split("(")[0]])
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", label="Full Switching", linestyle =  (5,(10,3)))
    objective_values_omega_ratios = {}
    for o in omega:
        for cmin in objective_values_omega[o].keys():
            for key in objective_values_omega[o][cmin].keys():
                if o not in objective_values_omega_ratios.keys():
                    if objective_values_no_switch[cmin][key] > 0.0000001:
                        objective_values_omega_ratios[o] ={cmin: [objective_values_omega[o][cmin][key] / objective_values_no_switch[cmin][key]]}
                else:
                    if cmin not in objective_values_omega_ratios[o].keys():
                        if objective_values_no_switch[cmin][key] > 0.0000001:
                            objective_values_omega_ratios[o][cmin] = [
                                objective_values_omega[o][cmin][key] / objective_values_no_switch[cmin][key]]
                    else:
                        if objective_values_no_switch[cmin][key] > 0.0000001:
                            objective_values_omega_ratios[o][cmin].extend(
                                [objective_values_omega[o][cmin][key] / objective_values_no_switch[cmin][key]])
    for o in omega:
        mean_objectives = {}
        std_objectives = {}
        for key in  objective_values_omega_ratios[o].keys():
            mean_objectives[key] = np.mean(objective_values_omega_ratios[o][key])
            std_objectives[key] = np.std(objective_values_omega_ratios[o][key])
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
        linestyle = ["-", "--", ":", "-.", (5,(10,3))]
        # cross_val_scores[fun]= cross_val_score(y_values = mean_differences, x_values = x, k = 5, model = getsource(fun).split(" ")[1].split("(")[0]])
        plt.errorbar(x, mean_differences, yerr=std_differences, label=f"Switching limited to {o} devices", linestyle = linestyle[o-2])
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
    plt.axhline(y=1, color='b', linestyle='-', label="Cost of Network without Switching")
    plt.xlabel("Minimum Capacity Necessary: cmin (bits/s)", fontsize=10)
    plt.ylabel("Cost of Network/Cost of Network Without Switching", fontsize=10)
    plt.legend(loc='upper left', fontsize='small')
    # plt.legend(loc='upper right', fontsize='small')
    plt.savefig("cmin_mesh_topology_comparison_different_number_switching_terms_bw.png")
    plt.show()





def plot_solution_for_varying_detector_costs_with_omega(hot_bob_capacity_file, cold_bob_capacity_file, hot_bob_capacity_file_no_switch, cold_bob_capacity_file_no_switch, N, M,  cmin, cost_on_h, cost_on_c, f_switch, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_no_switch = None):
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file,
        cold_bob_capacity_file=cold_bob_capacity_file, cmin=1000)

    hot_capacity_dict_no_switch, cold_capacity_dict_no_switch, required_connections_no_switch, distances_no_switch = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file=hot_bob_capacity_file_no_switch,
        cold_bob_capacity_file=cold_bob_capacity_file_no_switch, cmin=1000)
    # graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["link cost"]
            current_key = last_row_explored["Graph key"].iloc[0]
            link_cost_current = last_ratio_done.iloc[0]
        else:
            link_cost_current = None
            current_key = None
            dictionary_fieldnames = ["link cost", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        link_cost_current = None
        current_key = None
    no_soln_set = []
    objective_values_link_cost = {}
    lambda_values = {}
    for link_cost in np.arange(start=10, stop=105, step=5):
        if link_cost_current != None:
            if link_cost != link_cost_current:
                continue
            else:
                link_cost_current = None
        for key in hot_capacity_dict.keys():
            if key not in no_soln_set:
                if current_key != key and current_key != None:
                    continue
                elif current_key == key:
                    current_key = None
                    continue
                uncooled_cost = (25 + link_cost) / 125
                cooled_cost  = (40 + link_cost) / 125
                try:
                    for req in required_connections[key].keys():
                        required_connections[key][req] = float(cmin)
                    sol_dict, prob, time_to_solve = optimisation_switching_model.initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict=hot_capacity_dict[key],
                        cold_key_dict=cold_capacity_dict[key],
                        required_connections=required_connections[key],
                        cost_det_h=uncooled_cost, cost_det_c=cooled_cost,
                        cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N, M=M,
                        fract_switch=f_switch,
                        time_limit=1e3, early_stop=0.003)
                except:
                    no_soln_set.append(key)
                    continue
                if optimisation_switched.check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if link_cost not in objective_values_link_cost.keys():
                        objective_values_link_cost[link_cost] = {key: objective_value}
                    else:
                        objective_values_link_cost[link_cost][key] = objective_value
                    # print results for on nodes:
                    use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = optimisation_switching_model.split_sol_dict(sol_dict)
                    for lambda_key in lambda_dict.keys():
                        if lambda_key not in lambda_values.keys():
                            lambda_values[lambda_key] = {link_cost: lambda_dict[lambda_key]}
                        else:
                            lambda_values[lambda_key][link_cost] = lambda_dict[lambda_key]
                    print("Results for link cost:" + str(link_cost))
                    for value in lambda_dict:
                        print("Value for " + str(value) + ": " + str(lambda_dict[value]))
                    for binary in binary_dict:
                        print("Value for " + str(binary) + ": " + str(binary_dict[binary]))
                    if data_storage_location_keep_each_loop != None:
                        dictionary = [
                            {"link cost": link_cost, "Graph key": key, "objective_value": objective_value}]
                        dictionary_fieldnames = ["link cost", "Graph key", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)

    if data_storage_location_keep_each_loop_no_switch != None:
        if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["link cost"]
            current_key = last_row_explored["Graph key"].iloc[0]
            link_cost_current = last_ratio_done.iloc[0]
        else:
            link_cost_current = None
            current_key = None
            dictionary_fieldnames = ["link cost", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        link_cost_current = None
        current_key = None
    no_soln_set = []
    objective_values_no_switch = {}
    lambda_values = {}
    for link_cost in np.arange(start=10, stop=105, step=5):
        if link_cost_current != None:
            if link_cost != link_cost_current:
                continue
            else:
                link_cost_current = None
        for key in hot_capacity_dict_no_switch.keys():
            if key not in no_soln_set:
                if current_key != key and current_key != None:
                    continue
                elif current_key == key:
                    current_key = None
                    continue
                try:
                    uncooled_cost = (25 + link_cost) / 125
                    cooled_cost = (40 + link_cost) / 125
                    for req in required_connections_no_switch[key].keys():
                        required_connections_no_switch[key][req] = float(cmin)
                    sol_dict, prob = optimisation_no_switching.initial_optimisation(
                        hot_key_dict=hot_capacity_dict_no_switch[key],
                        cold_key_dict=cold_capacity_dict_no_switch[key],
                        required_connections=required_connections_no_switch[key], cost_det_h=uncooled_cost,
                        cost_det_c=cooled_cost, cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N * 780, M=M)
                except:
                    no_soln_set.append(key)
                    continue
                if optimisation_switched.check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if link_cost not in objective_values_no_switch.keys():
                        objective_values_no_switch[link_cost] = {key: objective_value}
                    else:
                        objective_values_no_switch[link_cost][key] = objective_value
                    # print results for on nodes:
                    use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = optimisation_switching_model.split_sol_dict(sol_dict)
                    for lambda_key in lambda_dict.keys():
                        if lambda_key not in lambda_values.keys():
                            lambda_values[lambda_key] = {link_cost: lambda_dict[lambda_key]}
                        else:
                            lambda_values[lambda_key][link_cost] = lambda_dict[lambda_key]
                    print("Results for link cost:" + str(link_cost))
                    for value in lambda_dict:
                        print("Value for " + str(value) + ": " + str(lambda_dict[value]))
                    for binary in binary_dict:
                        print("Value for " + str(binary) + ": " + str(binary_dict[binary]))
                    if data_storage_location_keep_each_loop_no_switch != None:
                        dictionary = [
                            {"link cost": link_cost, "Graph key": key, "objective_value": objective_value}]
                        dictionary_fieldnames = ["link cost", "Graph key", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
                            with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)




    omega =[2,3,4,5]
    objective_values_omega = {}
    for o in omega:
        if data_storage_location_keep_each_loop != None:
            if os.path.isfile(data_storage_location_keep_each_loop + str(o) + '.csv'):
                plot_information = pd.read_csv(data_storage_location_keep_each_loop + str(o) + ".csv")
                last_row_explored = plot_information.iloc[[-1]]
                last_ratio_done = last_row_explored["link cost"]
                current_key = last_row_explored["Graph key"].iloc[0]
                link_cost_current = last_ratio_done.iloc[0]
            else:
                link_cost_current = None
                current_key = None
                dictionary_fieldnames = ["link cost", "Graph key", "objective_value"]
                with open(data_storage_location_keep_each_loop + str(o) + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                    writer.writeheader()
        else:
            link_cost_current = None
            current_key = None
        no_soln_set = []
        objective_values_link_costs = {}
        lambda_values = {}
        for link_cost in np.arange(start=10, stop=105, step=5):
            if link_cost_current != None:
                if link_cost != link_cost_current:
                    continue
                else:
                    link_cost_current = None
            for key in hot_capacity_dict.keys():
                if key not in no_soln_set:
                    if current_key != key and current_key != None:
                        continue
                    elif current_key == key:
                        current_key = None
                        continue
                    try:
                        uncooled_cost = (25 + link_cost) / 125
                        cooled_cost = (40 + link_cost) / 125
                        for req in required_connections[key].keys():
                            required_connections[key][req] = float(cmin)
                        sol_dict, prob, time_to_solve = initial_optimisation_switching_limited_number_of_switching(hot_key_dict=hot_capacity_dict[key],
                            cold_key_dict=cold_capacity_dict[key],
                            required_connections=required_connections[key],
                            cost_det_h=uncooled_cost, cost_det_c=cooled_cost,
                            cost_on_h=cost_on_h, cost_on_c=cost_on_c, N=N * 500, M=M,
                            omega = o,
                            fract_switch=f_switch,
                            time_limit=1e3, early_stop=0.003)
                    except:
                        no_soln_set.append(key)
                        continue
                    if optimisation_switched.check_solvable(prob):
                        objective_value = prob.solution.get_objective_value()
                        if o not in objective_values_omega.keys():
                            objective_values_omega[o] = {link_cost: {key: objective_value}}
                        elif link_cost not in objective_values_omega[o].keys():
                            objective_values_omega[o][link_cost] = {key: objective_value}
                        else:
                            objective_values_omega[o][link_cost][key] = objective_value
                        # print results for on nodes:
                        use_dict, lambda_dict, fraction_dict, binary_dict, detector_useage = optimisation_switching_model.split_sol_dict(
                            sol_dict)
                        for lambda_key in lambda_dict.keys():
                            if lambda_key not in lambda_values.keys():
                                lambda_values[lambda_key] = {link_cost: lambda_dict[lambda_key]}
                            else:
                                lambda_values[lambda_key][link_cost] = lambda_dict[lambda_key]
                        print("Results for link cost:" + str(link_cost))
                        for value in lambda_dict:
                            print("Value for " + str(value) + ": " + str(lambda_dict[value]))
                        for binary in binary_dict:
                            print("Value for " + str(binary) + ": " + str(binary_dict[binary]))
                        if data_storage_location_keep_each_loop != None:
                            dictionary = [
                                {"link cost": link_cost, "Graph key": key, "objective_value": objective_value}]
                            dictionary_fieldnames = ["link cost", "Graph key", "objective_value"]
                            if os.path.isfile(data_storage_location_keep_each_loop + str(o) + '.csv'):
                                with open(data_storage_location_keep_each_loop +str(o) + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writerows(dictionary)
                            else:
                                with open(data_storage_location_keep_each_loop + str(o) + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writeheader()
                                    writer.writerows(dictionary)


    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["link cost"] not in objective_values_link_cost.keys():
                objective_values_link_cost[row["link cost"]] = {row["Graph key"] : row["objective_value"]}
            else:
                objective_values_link_cost[row["link cost"]][row["Graph key"]] = row["objective_value"]
        for o in omega:
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + str(o) + ".csv")
            for index, row in plot_information.iterrows():
                if o not in objective_values_omega.keys():
                    objective_values_omega[o] = {row["link cost"]: {row["Graph key"] : row["objective_value"]}}
                elif row["link cost"] not in objective_values_omega[o].keys():
                    objective_values_omega[o][row["link cost"]] = {row["Graph key"]: row["objective_value"]}
                else:
                    objective_values_omega[o][row["link cost"]][row["Graph key"]]= row["objective_value"]
    if data_storage_location_keep_each_loop_no_switch != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
        for index, row in plot_information.iterrows():
            if row["link cost"] not in objective_values_no_switch.keys():
                objective_values_no_switch[row["link cost"]] = {row["Graph key"]: row["objective_value"]}
            else:
                objective_values_no_switch[row["link cost"]][row["Graph key"]] = row["objective_value"]
    objective_values_link_cost_ratios = {}
    for link_cost in objective_values_link_cost.keys():
        for key in objective_values_link_cost[link_cost].keys():
            if link_cost not in objective_values_link_cost_ratios.keys():
                if objective_values_no_switch[link_cost][key] > 0.0000001:
                    objective_values_link_cost_ratios[link_cost] = [objective_values_link_cost[link_cost][key]/objective_values_no_switch[link_cost][key]]
            else:
                if objective_values_no_switch[cmin][key] > 0.0000001:
                    objective_values_link_cost_ratios[cmin].extend([objective_values_link_cost[link_cost][key] / objective_values_no_switch[link_cost][key]])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values_link_cost_ratios.keys():
        mean_objectives[key] = np.mean(objective_values_link_cost_ratios[key])
        std_objectives[key] = np.std(objective_values_link_cost_ratios[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        cost_ratio = (25 + key)/ 125
        x.append(cost_ratio)
    functions = {}
    y_predictions = {}
        # cross_val_scores[fun]= cross_val_score(y_values = mean_differences, x_values = x, k = 5, model = getsource(fun).split(" ")[1].split("(")[0]])
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", label="Full Switching", linestyle =  (5,(10,3)))
    objective_values_omega_ratios = {}
    for o in omega:
        for link_cost in objective_values_omega[o].keys():
            for key in objective_values_omega[o][link_cost].keys():
                if o not in objective_values_omega_ratios.keys():
                    if objective_values_no_switch[link_cost][key] > 0.0000001:
                        objective_values_omega_ratios[o] ={link_cost: [objective_values_omega[o][link_cost][key] / objective_values_no_switch[link_cost][key]]}
                else:
                    if link_cost not in objective_values_omega_ratios[o].keys():
                        if objective_values_no_switch[link_cost][key] > 0.0000001:
                            objective_values_omega_ratios[o][link_cost] = [
                                objective_values_omega[o][link_cost][key] / objective_values_no_switch[link_cost][key]]
                    else:
                        if objective_values_no_switch[link_cost][key] > 0.0000001:
                            objective_values_omega_ratios[o][link_cost].extend(
                                [objective_values_omega[o][link_cost][key] / objective_values_no_switch[link_cost][key]])
    for o in omega:
        mean_objectives = {}
        std_objectives = {}
        for key in  objective_values_omega_ratios[o].keys():
            mean_objectives[key] = np.mean(objective_values_omega_ratios[o][key])
            std_objectives[key] = np.std(objective_values_omega_ratios[o][key])
        mean_differences = []
        std_differences = []
        # topologies
        x = []
        for key in mean_objectives.keys():
            mean_differences.append(mean_objectives[key])
            std_differences.append(std_objectives[key])
            cost_ratio = (25 + key) / 125
            x.append(cost_ratio)
        functions = {}
        y_predictions = {}
        linestyle = ["-", "--", ":", "-.", (5,(10,3))]
        # cross_val_scores[fun]= cross_val_score(y_values = mean_differences, x_values = x, k = 5, model = getsource(fun).split(" ")[1].split("(")[0]])
        plt.errorbar(x, mean_differences, yerr=std_differences, label=f"Switching limited to {o} devices", linestyle = linestyle[o-2])
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
    plt.axhline(y=1, color='b', linestyle='-', label="Cost of Network without Switching")
    plt.xlabel("Cost of Detectors Scale Factor", fontsize=12)
    plt.ylabel("Cost of Network with Switching/Cost of Network Without Switching", fontsize=10)
    plt.legend(loc='upper left', fontsize='small')
    # plt.legend(loc='upper right', fontsize='small')
    plt.savefig("link_cost_mesh_topology_comparison_different_number_switching_terms_bw.png")
    plt.show()



if __name__ == "__main__":
    # plot_solution_for_varying_capacity_with_omega(hot_bob_capacity_file = "10_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file= "10_nodes_mesh_topology_35_cold_capacity",
    #                                                   hot_bob_capacity_file_no_switch= "10_nodes_mesh_topology_35_hot_capacity_no_switch", cold_bob_capacity_file_no_switch="10_nodes_mesh_topology_35_cold_capacity_no_switch", N = 1200,
    #                                                   M = 2, cost_det_h = 1, cost_det_c = 1.136, cost_on_h = 1.68, cost_on_c = 3.27, f_switch = 0.1,
    #                                               data_storage_location_keep_each_loop = "cmin_parameter_sweep_omega_values",
    #                                               data_storage_location_keep_each_loop_no_switch="cmin_parameter_sweep_omega_values_no_switching")
    plot_solution_for_varying_detector_costs_with_omega(hot_bob_capacity_file="10_nodes_mesh_topology_35_hot_capacity",
                                                  cold_bob_capacity_file="10_nodes_mesh_topology_35_cold_capacity",
                                                  hot_bob_capacity_file_no_switch="10_nodes_mesh_topology_35_hot_capacity_no_switch",
                                                  cold_bob_capacity_file_no_switch="10_nodes_mesh_topology_35_cold_capacity_no_switch",
                                                  N=12,
                                                  M=2, cmin = 1000, cost_on_h=1.68, cost_on_c=3.27,
                                                  f_switch=0.1,
                                                  data_storage_location_keep_each_loop="link_cost_parameter_sweep_omega_values",
                                                  data_storage_location_keep_each_loop_no_switch="link_cost_parameter_sweep_omega_values_no_switching")

    # test_differences(hot_bob_capacity_file = "10_nodes_mesh_topology_35_hot_capacity", cold_bob_capacity_file= "10_nodes_mesh_topology_35_cold_capacity",
    #                                                   hot_bob_capacity_no_switching= "10_nodes_mesh_topology_35_hot_capacity_no_switch", cold_bob_capacity_no_switching ="10_nodes_mesh_topology_35_cold_capacity_no_switch", N = 1200,
    #                                                   M = 2, cost_det_h = 1, cost_det_c = 1.136, cost_on_h = 1.68, cost_on_c = 3.27, f_switch = 0.1, cmin =1000,
    #                                                   data_storage_location_keep_each_loop="cmin_parameter_sweep",
    #                         .                          data_storage_location_no_switching="cmin_parameter_sweep_no_switching")

