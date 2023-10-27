import copy

from LP_relaxation import *
import numpy as np
from copy import deepcopy
import time

def get_int_of_key(key):
    new_string = key.split("_")[1]
    return int(new_string.split("^")[0])


def remove_detector_node(hot_capacity_dict, cold_capacity_dict, detector_to_remove):
    new_hot_capacity_dict = {}
    new_cold_capacity_dict = {}
    if detector_to_remove[1] == "h":
        for key in hot_capacity_dict.keys():
            if key[2] != detector_to_remove[0]:
                new_hot_capacity_dict[key] = hot_capacity_dict[key]
        new_cold_capacity_dict = copy.deepcopy(cold_capacity_dict)
    elif detector_to_remove[1] == "c":
        for key in cold_capacity_dict.keys():
            if key[2] != detector_to_remove[0]:
                new_cold_capacity_dict[key] = cold_capacity_dict[key]
        new_hot_capacity_dict = copy.deepcopy(hot_capacity_dict)
    return new_hot_capacity_dict, new_cold_capacity_dict

class Relaxed_Heuristic():

    def __init__(self, c_det_hot, c_det_cold, c_cold_on, c_hot_on, f_switch, Lambda, M):
        self.c_det_hot = c_det_hot
        self.c_det_cold = c_det_cold
        self.c_cold_on = c_cold_on
        self.c_hot_on = c_hot_on
        self.f_switch = f_switch
        self.Lambda = Lambda
        self.M = M

    def calculate_current_solution_cost(self, model):
        cost = 0.0
        detectors_hot = {}
        detectors_cold = {}
        for k in model.hot_key_dict:
            if not detectors_hot:
                detectors_hot[k[2]] = 1
            elif k[2] not in detectors_hot:
                detectors_hot[k[2]] = 1
        for k in model.cold_key_dict:
            if not detectors_cold:
                detectors_cold[k[2]] = 1
            elif k[2] not in detectors_cold:
                detectors_cold[k[2]] = 1

        for key in detectors_cold:
            lambda_var = f"lambda_{key}^c"
            if model.model.get_var_by_name(lambda_var).solution_value > 0.00000001:
                cost += self.c_cold_on
                cost += np.ceil(model.model.get_var_by_name(lambda_var).solution_value) * self.c_det_cold
        for key in detectors_hot:
            lambda_var = f"lambda_{key}^h"
            if model.model.get_var_by_name(lambda_var).solution_value > 0.00000001:
                cost += self.c_hot_on
                cost += np.ceil(model.model.get_var_by_name(lambda_var).solution_value) * self.c_det_hot


        # for key in model.model.cold_capacity_detectors.keys():
        #     if model.model.cold_capacity_detectors[key].solution_value > 0.00000001:
        #         cost += self.c_cold_on
        #         cost += np.ceil(model.model.cold_capacity_detectors[key].solution_value) * self.c_det_cold
        # for key in model.model.hot_capacity_detectors.keys():
        #     if model.model.hot_capacity_detectors[key].solution_value > 0.00000001:
        #         cost += self.c_hot_on
        #         cost += np.ceil(model.model.hot_capacity_detectors[key].solution_value) * self.c_det_hot
        return cost


    def calculate_difference_in_cost(self, model_1, model_2):
        cost_model_1 = self.calculate_current_solution_cost(model_1)
        cost_model_2 = self.calculate_current_solution_cost(model_2)
        return cost_model_1 - cost_model_2

    def remove_detectors_not_in_use(self, model):
        detectors_to_remove_hot = {}
        detectors_to_remove_cold = {}
        for k in model.hot_key_dict:
            if not detectors_to_remove_hot:
                detectors_to_remove_hot[(k[2], "h")] = 1
            elif k[2] not in detectors_to_remove_hot:
                detectors_to_remove_hot[(k[2], "h")] = 1
        for k in model.cold_key_dict:
            if not detectors_to_remove_cold:
                detectors_to_remove_cold[(k[2], "c")] = 1
            elif k[2] not in detectors_to_remove_cold:
                detectors_to_remove_cold[(k[2], "c")] = 1
        detectors_cold = copy.deepcopy(detectors_to_remove_cold)
        detectors_hot = copy.deepcopy(detectors_to_remove_hot)
        for key in detectors_cold:
            lambda_var = f"lambda_{key[0]}^c"
            if model.model.get_var_by_name(lambda_var).solution_value > 0.00000001 and key in detectors_to_remove_cold.keys():
                detectors_to_remove_cold.pop(key)
        for key in detectors_hot:
            lambda_var = f"lambda_{key[0]}^h"
            if model.model.get_var_by_name(lambda_var).solution_value > 0.00000001 and key in detectors_to_remove_hot.keys():
                detectors_to_remove_hot.pop(key)
        new_hot_capacity_dict = deepcopy(model.hot_key_dict)
        new_cold_capacity_dict = deepcopy(model.cold_key_dict)
        for key in detectors_to_remove_hot.keys():
            new_hot_capacity_dict, new_cold_capacity_dict = remove_detector_node(new_hot_capacity_dict, new_cold_capacity_dict, key)
        for key in detectors_to_remove_cold.keys():
            new_hot_capacity_dict, new_cold_capacity_dict = remove_detector_node(new_hot_capacity_dict, new_cold_capacity_dict, key)



        # for key in model.model.cold_capacity_detectors.keys():
        #     if model.model.cold_capacity_detectors[key].solution_value > 0.00000001 and (get_int_of_key(key), "c") in detectors_to_remove_hot:
        #         detectors_to_remove_hot.pop((get_int_of_key(key), "c"))
        # for key in model.model.hot_capacity_detectors.keys():
        #     if model.model.hot_capacity_detectors[key].solution_value > 0.00000001 and (get_int_of_key(key), "h") in detectors_to_remove_hot:
        #         detectors_to_remove_hot.pop((get_int_of_key(key), "h"))
        # new_hot_capacity_dict = deepcopy(model.hot_key_dict)
        # new_cold_capacity_dict = deepcopy(model.cold_key_dict)
        # for key in detectors_to_remove_hot.keys():
        #     new_hot_capacity_dict, new_cold_capacity_dict = remove_detector_node(new_hot_capacity_dict, new_cold_capacity_dict, key)
        return new_hot_capacity_dict, new_cold_capacity_dict

    def single_step_down(self, model):
        new_hot_capacity_dict, new_cold_capacity_dict = self.remove_detectors_not_in_use(model)
        detectors = {}
        for k in new_hot_capacity_dict:
            if not detectors:
                detectors[(k[2], "h")] = 1
            elif k[2] not in detectors:
                detectors[(k[2], "h")] = 1
        for k in new_cold_capacity_dict:
            if not detectors:
                detectors[(k[2], "c")] = 1
            elif k[2] not in detectors:
                detectors[(k[2], "c")] = 1
        # if reducing the problem by the size of one detector means there are not enough detector solutions to ensure
        # the multipath condition then the model is optimal, return it and its solution. Equal sign is because we have
        # yet to start removing the detectors in use - thus if we remove them we go to len(detectors)-1 detector sites.
        if len(detectors) <= self.M:
            return model, True
        else:
            # now remove the detectors in use (in detectors dictionary) one by one and calculate the new models

            models = []
            for key in detectors.keys():
                new_hot_capacity_dict_key, new_cold_capacity_dict_key = remove_detector_node(new_hot_capacity_dict, new_cold_capacity_dict, detector_to_remove=key)
                new_model = LP_Switched_fixed_switching_time_relaxation(name=model.model.name,
                                                                        hot_key_dict=new_hot_capacity_dict_key,
                                                                        cold_key_dict=new_cold_capacity_dict_key,
                                                                        Lambda=self.Lambda,
                                                                        required_connections=model.required_connections)
                new_model.set_up_problem(M= self.M, f_switch = self.f_switch, cost_det_h = self.c_det_hot, cost_det_c = self.c_det_cold)
                if new_model.model.solve():
                    obj = new_model.model.objective_value
                    models.append(new_model)
            current_best_model = None
            current_cost_improvement = 0.0
            for model_new in models:
                cost_improvement = self.calculate_difference_in_cost(model_new, model)
                if cost_improvement < current_cost_improvement:
                    current_best_model = model_new
                    current_cost_improvement = cost_improvement
            if current_best_model == None:
                # in this case all viable solutions of removing a node yield an increase in cost and thus the most
                # cost effective solution is to use the solution of the model:
                return model, True
            else:
                # otherwise there is an improved model - return the best improved model and the fact that we haven't
                # finished yet
                return current_best_model, False

    def single_step_down_faster(self, model):
        new_hot_capacity_dict, new_cold_capacity_dict = self.remove_detectors_not_in_use(model)
        detectors = {}
        for k in new_hot_capacity_dict:
            if not detectors:
                detectors[(k[2], "h")] = 1
            elif k[2] not in detectors:
                detectors[(k[2], "h")] = 1
        for k in new_cold_capacity_dict:
            if not detectors:
                detectors[(k[2], "c")] = 1
            elif k[2] not in detectors:
                detectors[(k[2], "c")] = 1
        # if reducing the problem by the size of one detector means there are not enough detector solutions to ensure
        # the multipath condition then the model is optimal, return it and its solution. Equal sign is because we have
        # yet to start removing the detectors in use - thus if we remove them we go to len(detectors)-1 detector sites.
        if len(detectors) <= self.M:
            return model, True
        else:
            # now remove the detectors in use (in detectors dictionary) one by one and calculate the new models
            new_model = LP_Switched_fixed_switching_time_relaxation(name=model.model.name,
                                                                    hot_key_dict=new_hot_capacity_dict,
                                                                    cold_key_dict=new_cold_capacity_dict,
                                                                    Lambda=self.Lambda,
                                                                    required_connections=model.required_connections)
            new_model.set_up_problem(M=self.M, f_switch=self.f_switch, cost_det_h=self.c_det_hot,
                                     cost_det_c=self.c_det_cold)
            models = []
            for key in detectors.keys():
                current_model = new_model.clone()
                current_model.set_detector_node_off(key)
                if current_model.model.solve():
                    obj = current_model.model.objective_value
                    models.append(current_model)
            current_best_model = None
            current_cost_improvement = 0.0
            for model_new in models:
                cost_improvement = self.calculate_difference_in_cost(model_new, model)
                if cost_improvement < current_cost_improvement:
                    current_best_model = model_new
                    current_cost_improvement = cost_improvement
            if current_best_model == None:
                # in this case all viable solutions of removing a node yield an increase in cost and thus the most
                # cost effective solution is to use the solution of the model:
                return model, True
            else:
                # otherwise there is an improved model - return the best improved model and the fact that we haven't
                # finished yet
                return current_best_model, False



    def full_recursion(self, initial_model):
        initial_model.set_up_problem(M= self.M, f_switch = self.f_switch, cost_det_h = self.c_det_hot, cost_det_c = self.c_det_cold)
        if initial_model.model.solve():
            model = initial_model
            not_complete = True
            while not_complete:
                new_best_model, complete = self.single_step_down(model)
                model = new_best_model
                not_complete = not complete
            return model

    def full_recursion_faster(self, initial_model):
        initial_model.set_up_problem(M= self.M, f_switch = self.f_switch, cost_det_h = self.c_det_hot, cost_det_c = self.c_det_cold)
        if initial_model.model.solve():
            model = initial_model
            not_complete = True
            while not_complete:
                new_best_model, complete = self.single_step_down_faster(model)
                model = new_best_model
                not_complete = not complete
            return model


if __name__ == "__main__":
    hot_capacity_dict, cold_capacity_dict, required_connections, distances = import_switched_network_values_multiple_graphs(
        hot_bob_capacity_file="1_nodes_mesh_topology_35_hot_capacity",
        cold_bob_capacity_file="1_nodes_mesh_topology_35_cold_capacity", cmin=1000)
    for i in range(0, 10):
        model = LP_Switched_fixed_switching_time_relaxation(name=f"problem_{i}", hot_key_dict=hot_capacity_dict[i],
                                                            cold_key_dict=cold_capacity_dict[i],
                                                            Lambda=12,
                                                            required_connections=required_connections[i])
        heuristic = Relaxed_Heuristic(c_det_hot = 1, c_det_cold = 2, c_cold_on = 4, c_hot_on = 2, Lambda=12, f_switch = 0.15, M=2)
        # t_0 = time.time()
        # model_best = heuristic.full_recursion_faster(initial_model=model)
        t_1 = time.time()
        # print("Time for faster recursion " + str(t_1-t_0))
        # try:
        #     print(str(heuristic.calculate_current_solution_cost(model_best)))
        # except:
        #     print("No solution")
        #     continue
        model_best = heuristic.full_recursion(initial_model= model)
        t_2 = time.time()
        print("Time for recursion " + str(t_2 - t_1))
        try:
            print(str(heuristic.calculate_current_solution_cost(model_best)))
        except:
            print("No solution")
            continue