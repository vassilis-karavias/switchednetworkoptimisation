from docplex.mp.model import Model
import numpy as np
from switched_network_utils import *
from copy import deepcopy


class LP_Switched_fixed_switching_time_relaxation():

    def __init__(self, name,  hot_key_dict, cold_key_dict, required_connections, Lambda):
        self.model = Model(name = name)
        self.hot_key_dict = hot_key_dict
        self.cold_key_dict = cold_key_dict
        self.required_connections = required_connections
        fraction_capacity_variables_hot = {}
        fraction_capacity_variables_cold = {}
        detectors_hot = {}
        detectors_cold = {}
        for k in self.hot_key_dict:
            if not detectors_hot:
                detectors_hot[k[2]] = 1
            elif k[2] not in detectors_hot:
                detectors_hot[k[2]] = 1
        for k in self.cold_key_dict:
            if not detectors_cold:
                detectors_cold[k[2]] = 1
            elif k[2] not in detectors_cold:
                detectors_cold[k[2]] = 1
        for k in self.hot_key_dict:
            if not fraction_capacity_variables_hot:
                fraction_capacity_variables_hot[k] = 1
            elif k not in fraction_capacity_variables_hot:
                fraction_capacity_variables_hot[k] = 1
        for k in self.cold_key_dict:
            if not fraction_capacity_variables_cold:
                fraction_capacity_variables_cold[k] = 1
            elif k not in fraction_capacity_variables_cold:
                fraction_capacity_variables_cold[k] = 1
        capacity_variables_hot = []
        capacity_variables_cold = []
        for key in fraction_capacity_variables_hot:
            capacity_variables_hot.append(f"q_{key[0], key[1], key[2]}^h")
        for key in fraction_capacity_variables_cold:
            capacity_variables_cold.append(f"q_{key[0], key[1], key[2]}^c")
        self.model.hot_capacity_variables = self.model.continuous_var_dict(keys=capacity_variables_hot, lb=0.0, name = capacity_variables_hot)
        self.model.cold_capacity_variables = self.model.continuous_var_dict(keys=capacity_variables_cold, lb=0.0, name = capacity_variables_cold)
        # define the \lambda_{d}^{m} variables and add them to the list of variables
        self.Lambda = Lambda
        capacity_detectors_hot = []
        capacity_detectors_cold = []
        detectors_hot = []
        detectors_cold = []
        for key in fraction_capacity_variables_hot:
            if key[2] not in detectors_hot:
                capacity_detectors_hot.append(f"lambda_{key[2]}^h")
                detectors_hot.append(key[2])
        for key in fraction_capacity_variables_cold:
            if key[2] not in detectors_cold:
                capacity_detectors_cold.append(f"lambda_{key[2]}^c")
                detectors_cold.append(key[2])
        self.model.hot_capacity_detectors = self.model.continuous_var_dict(keys=capacity_detectors_hot, lb=0.0,
                                                                           ub=Lambda, name = capacity_detectors_hot)
        self.model.cold_capacity_detectors = self.model.continuous_var_dict(keys=capacity_detectors_cold, lb=0.0,
                                                                            ub=Lambda, name = capacity_detectors_cold)


    def add_minimum_capacity_constraint(self, M):
        fraction_capacity_variables_hot = {}
        fraction_capacity_variables_cold = {}
        detectors_hot = {}
        detectors_cold = {}
        for k in self.hot_key_dict:
            if not detectors_hot:
                detectors_hot[k[2]] = 1
            elif k[2] not in detectors_hot:
                detectors_hot[k[2]] = 1
        for k in self.cold_key_dict:
            if not detectors_cold:
                detectors_cold[k[2]] = 1
            elif k[2] not in detectors_cold:
                detectors_cold[k[2]] = 1
        for k in self.hot_key_dict:
            if not fraction_capacity_variables_hot:
                fraction_capacity_variables_hot[k] = 1
            elif k not in fraction_capacity_variables_hot:
                fraction_capacity_variables_hot[k] = 1
        for k in self.cold_key_dict:
            if not fraction_capacity_variables_cold:
                fraction_capacity_variables_cold[k] = 1
            elif k not in fraction_capacity_variables_cold:
                fraction_capacity_variables_cold[k] = 1
        for connection in self.required_connections:
            capacities_hot = {}
            capacities_cold = {}
            for detector in detectors_hot:
                capacities_hot[f"q_{connection[0],connection[1],detector}^h"]= float(self.hot_key_dict[(connection[0], connection[1], detector)])
            for detector in detectors_cold:
                capacities_cold[f"q_{connection[0],connection[1],detector}^c"]= float(self.cold_key_dict[(connection[0], connection[1], detector)])
            self.model.total_capacity_hot = self.model.sum(self.model.hot_capacity_variables[key] * capacities_hot[key] for key in capacities_hot.keys())
            self.model.total_capacity_cold = self.model.sum(self.model.cold_capacity_variables[key] * capacities_cold[key] for key in capacities_cold.keys())
            self.model.add_constraint(self.model.total_capacity_cold + self.model.total_capacity_hot >= M * self.required_connections[connection])

    def add_max_capacity_constraint(self, f_switch):
        """
        adds the constraint \sum_{i,j \in S} q_{k=(i,j,d)}^{m} \leq (1-f_switch) \lambda_{d}^{m}
        """
        # get a dictionary with keys the detectors and a dictionary with (i,j,d)
        fraction_capacity_variables_hot = {}
        fraction_capacity_variables_cold = {}
        detectors_hot = {}
        detectors_cold = {}
        for k in self.hot_key_dict:
            if not detectors_hot:
                detectors_hot[k[2]] = 1
            elif k[2] not in detectors_hot:
                detectors_hot[k[2]] = 1
        for k in self.cold_key_dict:
            if not detectors_cold:
                detectors_cold[k[2]] = 1
            elif k[2] not in detectors_cold:
                detectors_cold[k[2]] = 1
        for k in self.hot_key_dict:
            if not fraction_capacity_variables_hot:
                fraction_capacity_variables_hot[k] = 1
            elif k not in fraction_capacity_variables_hot:
                fraction_capacity_variables_hot[k] = 1
        for k in self.cold_key_dict:
            if not fraction_capacity_variables_cold:
                fraction_capacity_variables_cold[k] = 1
            elif k not in fraction_capacity_variables_cold:
                fraction_capacity_variables_cold[k] = 1
        # insert constraint
        lambda_cold = []
        lambda_hot = []
        q_cold = []
        q_hot = []
        new_detectors_hot = []
        new_detectors_cold = []
        for key in fraction_capacity_variables_hot:
            if key[2] not in new_detectors_hot:
                lambda_hot.append(f"lambda_{key[2]}^h")
                new_detectors_hot.append(key[2])
                q_hot.append([f"q_{key[0], key[1], key[2]}^h"])
            else:
                for i in range(len(new_detectors_hot)):
                    if key[2] == new_detectors_hot[i]:
                        q_hot[i].append(f"q_{key[0], key[1], key[2]}^h")
        for key in fraction_capacity_variables_cold:
            if key[2] not in new_detectors_cold:
                lambda_cold.append(f"lambda_{key[2]}^c")
                new_detectors_cold.append(key[2])
                q_cold.append([f"q_{key[0], key[1], key[2]}^c"])
            else:
                for i in range(len(new_detectors_cold)):
                    if key[2] == new_detectors_cold[i]:
                        q_cold[i].append(f"q_{key[0], key[1], key[2]}^c")
        for i in range(len(q_cold)):
            self.model.q_current_cold = self.model.sum(self.model.cold_capacity_variables[key] for key in q_cold[i])
            self.model.add_constraint(self.model.q_current_cold <= (1 - f_switch) * self.model.cold_capacity_detectors[lambda_cold[i]])
        for i in range(len(q_hot)):
            self.model.q_current_hot = self.model.sum(self.model.hot_capacity_variables[key] for key in q_hot[i])
            self.model.add_constraint(self.model.q_current_hot <= (1-f_switch) * self.model.hot_capacity_detectors[lambda_hot[i]])


    def add_maximal_single_path_capacity_multipath(self):
        """
        adds the constraint to prevent any individual path from having a capacity greater than c_{min} - this ensures that
        at least M separate paths need to be used to meet the capacity requirements: c_{k}^{m}q_{k}^{m} \leq \frac{c_{i,j}}{M} c_{min}
        """
        detectors_hot = {}
        detectors_cold = {}
        for k in self.hot_key_dict:
            if not detectors_hot:
                detectors_hot[k[2]] = 1
            elif k[2] not in detectors_hot:
                detectors_hot[k[2]] = 1
        for k in self.cold_key_dict:
            if not detectors_cold:
                detectors_cold[k[2]] = 1
            elif k[2] not in detectors_cold:
                detectors_cold[k[2]] = 1
        for connection in self.required_connections:
            for detector in detectors_hot:
                ind_flow_hot = f"q_{connection[0], connection[1], detector}^h"
                capacity_hot = float(self.hot_key_dict[(connection[0], connection[1], detector)])
                if capacity_hot > 0.00001:
                    self.model.add_constraint(self.model.hot_capacity_variables[ind_flow_hot] * capacity_hot <= self.required_connections[connection])
            for detector in detectors_cold:
                ind_flow_cold = f"q_{connection[0], connection[1], detector}^c"
                capacity_cold = float(self.cold_key_dict[(connection[0], connection[1], detector)])
                if capacity_cold> 0.000001:
                    self.model.add_constraint(self.model.cold_capacity_variables[ind_flow_cold] * capacity_cold <= self.required_connections[connection])


    def add_objective_function(self, cost_det_h, cost_det_c):
        self.model.detector_cost = self.model.sum(self.model.hot_capacity_detectors[key] *cost_det_h for key in self.model.hot_capacity_detectors.keys()) +self.model.sum(self.model.cold_capacity_detectors[key] * cost_det_c for key in self.model.cold_capacity_detectors.keys())
        self.model.minimize(self.model.detector_cost)

    def set_up_problem(self, M, f_switch, cost_det_h, cost_det_c):
        self.add_minimum_capacity_constraint(M)
        self.add_max_capacity_constraint(f_switch)
        self.add_maximal_single_path_capacity_multipath()
        self.add_objective_function(cost_det_h, cost_det_c)

    def set_detector_node_off(self, detector_node):
        current_detector = detector_node[0]
        temperature = detector_node[1]
        lambda_var = f"lambda_{current_detector}^{temperature}"
        self.model.add_constraint(self.model.get_var_by_name(lambda_var) == 0)
        for connection in self.required_connections:
            ind_flow_hot = f"q_{connection[0], connection[1], current_detector}^{temperature}"
            self.model.add_constraint(self.model.get_var_by_name(ind_flow_hot) == 0)

    def clone(self):
        model = LP_Switched_fixed_switching_time_relaxation(name = self.model.name, hot_key_dict = deepcopy(self.hot_key_dict)
                                                            , cold_key_dict = deepcopy(self.cold_key_dict), Lambda= self.Lambda,
                                                            required_connections = deepcopy(self.required_connections))
        model.model = self.model.clone()
        return model






if  __name__ == "__main__":
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file="1_nodes_mesh_topology_35_hot_capacity",
        cold_bob_capacity_file="1_nodes_mesh_topology_35_cold_capacity", cmin=1000)
    for i in range(0,10):

        model = LP_Switched_fixed_switching_time_relaxation(name = f"problem_{i}", hot_key_dict= hot_capacity_dict[i], cold_key_dict=cold_capacity_dict[i], required_connections=required_connections[i])
        model.set_up_problem(M= 2, f_switch = 0.1, Lambda = 6, cost_det_h = 1, cost_det_c = 5)


        model.model.print_information()
        if model.model.solve():
            obj = model.model.objective_value
            # for key in model.model.cold_capacity_variables.keys():
            #     print(str(key) + " solution value:" + str(model.model.cold_capacity_variables[key].solution_value))
            # for key in model.model.hot_capacity_variables.keys():
            #     print(str(key) + " solution value:" + str(model.model.hot_capacity_variables[key].solution_value))
            for key in model.model.cold_capacity_detectors.keys():
                print(str(key) + " solution value:" + str(model.model.cold_capacity_detectors[key].solution_value))
            for key in model.model.hot_capacity_detectors.keys():
                print(str(key) + " solution value:" + str(model.model.hot_capacity_detectors[key].solution_value))