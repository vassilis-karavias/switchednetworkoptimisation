import graph_tool as gt
import pandas as pd
import copy
import numpy as np
### method for importing data

def import_switched_network_values_multiple_graphs(hot_bob_capacity_file, cold_bob_capacity_file, cmin):
    hot_bob_capacities = pd.read_csv(hot_bob_capacity_file + ".csv")
    cold_bob_capacities = pd.read_csv(cold_bob_capacity_file + ".csv")
    possible_ids = hot_bob_capacities["ID"].unique()
    hot_capacity_dict_multiple_graphs = {}
    cold_capacity_dict_multiple_graphs = {}
    required_connections_multiple_graphs = {}
    distances = {}
    for id in possible_ids:
        hot_bob_capacities_id = hot_bob_capacities[hot_bob_capacities["ID"] == id].drop(["ID"], axis = 1)
        cold_bob_capacities_id = cold_bob_capacities[cold_bob_capacities["ID"] == id].drop(["ID"], axis=1)
        hot_capacity_dict = {}
        for index, row in hot_bob_capacities_id.iterrows():
            hot_capacity_dict[(int(row["source"]), int(row["target"]), int(row["detector"]))] = row["capacity"]
            if id not in distances.keys():
                distances[id] = row["size"]
        cold_capacity_dict = {}
        for index, row in cold_bob_capacities_id.iterrows():
            cold_capacity_dict[(int(row["source"]), int(row["target"]), int(row["detector"]))] = row["capacity"]
        hot_capacity_dict_multiple_graphs[id] = hot_capacity_dict
        cold_capacity_dict_multiple_graphs[id] = cold_capacity_dict

        required_connections = {}
        for index, row in hot_bob_capacities_id.iterrows():
            if not required_connections:
                required_connections[(int(row["source"]), int(row["target"]))] = cmin
            elif (int(row["source"]), int(row["target"])) not in required_connections:
                required_connections[(int(row["source"]), int(row["target"]))] = cmin
        for index, row in cold_bob_capacities_id.iterrows():
            if not required_connections:
                required_connections[(int(row["source"]), int(row["target"]))] = cmin
            elif (int(row["source"]), int(row["target"])) not in required_connections:
                required_connections[(int(row["source"]), int(row["target"]))] = cmin

        required_connections_multiple_graphs[id] = required_connections
        print("Finished: " + str(id))
    return hot_capacity_dict_multiple_graphs, cold_capacity_dict_multiple_graphs, required_connections_multiple_graphs, distances


class Heuristic_Model():

    def __init__(self, hot_capacity_dict, cold_capacity_dict, required_connections, C_node_cold, C_node_hot, C_det_cold, C_det_hot, M):
        # define all the initial parameters and fix the sets to their initial values as required by heuristic
        self.connection_pairs = set(required_connections.keys())
        self.required_connections = required_connections
        self.unsearched_pairs = copy.deepcopy(self.connection_pairs)
        self.searched_pairs = set()
        self.C_node_cold = C_node_cold
        self.C_node_hot = C_node_hot
        self.C_det_cold = C_det_cold
        self.C_det_hot = C_det_hot
        self.hot_capacity_dict = hot_capacity_dict
        self.cold_capacity_dict = cold_capacity_dict
        self.C_tot = 0.0
        self.mu = -M
        self.D_used_ij = {}
        for key in self.connection_pairs:
            self.D_used_ij[key] = set()
        # split the graph into detector nodes and source node sets - will be needed for ease of access of terms later
        self.detector_nodes = set()
        self.source_node = set()
        # cap path set:
        self.CAP_ij = {}
        for source, target, detector in self.cold_capacity_dict.keys():
            if source not in self.source_node:
                self.source_node.add(source)
            if target not in self.source_node:
                self.source_node.add(target)
            if detector not in self.detector_nodes:
                self.detector_nodes.add((detector, "c"))
                self.detector_nodes.add((detector, "h"))
            if (source,target) in self.CAP_ij.keys():
                self.CAP_ij[(source,target)][(detector, "h")] = self.hot_capacity_dict[(source,target,detector)]
                self.CAP_ij[(source,target)][(detector, "c")] = self.cold_capacity_dict[(source,target,detector)]
            else:
                self.CAP_ij[(source,target)] = {(detector, "c") : self.cold_capacity_dict[(source,target,detector)], (detector, "h"): self.hot_capacity_dict[(source, target, detector)]}
        self.on_detectors = set()
        self.off_detectors = copy.deepcopy(self.detector_nodes)
        # set parameters for detectors on node positions.
        self.N_d = {}
        self.alpha_d = {}
        for detector in self.detector_nodes:
            self.N_d[detector] = 0.0
            self.alpha_d[detector] = 0.0
        # need to sort the eleements of CAP_ij[(source,target)] by decreasing capacity
        for key in self.CAP_ij.keys():
            self.CAP_ij[key] = dict(sorted(self.CAP_ij[key].items(), key=lambda item: item[1], reverse=True))
        # get current_pair
        self.current_pair = None
        for key in self.CAP_ij.keys():
            if self.current_pair == None:
                self.current_pair = key
            else:
                if self.CAP_ij[key][list(self.CAP_ij[key].keys())[0]] <= self.CAP_ij[self.current_pair][list(self.CAP_ij[self.current_pair].keys())[0]]:
                    self.current_pair  = key


    def check_feasible(self):
        # test the condition for stopping and returning INFEASIBLE
        for key in self.CAP_ij.keys():
            if self.CAP_ij[key][list(self.CAP_ij[key].keys())[-self.mu -1]] <= 0.00001:
                return False
        return True

    def split_cap_ij(self):
        self.CAP_N = {}
        self.CAP_F = {}
        for key in self.CAP_ij[self.current_pair].keys():
            if key in self.on_detectors.difference(self.D_used_ij[self.current_pair]):
                self.CAP_N[key] = self.CAP_ij[self.current_pair][key]
            elif key in self.off_detectors:
                self.CAP_F[key] = self.CAP_ij[self.current_pair][key]


    def get_N(self, d_m):
        # return 1
        N = 0
        c_det = 0
        if d_m[1] == "c":
            c_det = self.C_det_cold
        elif d_m[1] == "h":
            c_det = self.C_det_hot
        for key in self.unsearched_pairs:
            gamma_key = 1
            for detector in self.CAP_N.keys():
                if detector[1] == "c":
                    if (self.required_connections[key]/self.CAP_ij[key][d_m]) * c_det >= (self.required_connections[key]/self.CAP_N[detector]) * self.C_det_cold:
                        gamma_key = 0
                elif detector[1] == "h":
                    if (self.required_connections[key]/self.CAP_ij[key][d_m]) * c_det >= (self.required_connections[key]/self.CAP_N[detector]) * self.C_det_hot:
                        gamma_key = 0
                else:
                    raise ValueError
            N += gamma_key
        if N < 1:
            return 1
        else:
            return N

    def lowest_current_cost_detector(self):
        # find the lowest cost & detector for the set CAP_N:
        C_min_N = np.infty
        d_min_N = None
        for detector, temp in self.CAP_N.keys():
            if self.CAP_N[(detector,temp)] > 0.00001:
                if temp == "c":
                    C_curr = (self.required_connections[self.current_pair] / self.CAP_N[(detector, temp)]) * self.C_det_cold
                        # max(np.ceil((self.required_connections[self.current_pair] / self.CAP_N[(detector, temp)]) - self.alpha_d[(detector, temp)]) * self.C_det_cold - (1 - (self.required_connections[self.current_pair]/self.CAP_N[d_min_N] - self.alpha_d[d_min_N])%1) * self.C_det_cold, 0)
                elif temp == "h":
                    C_curr = (self.required_connections[self.current_pair] / self.CAP_N[(detector, temp)]) * self.C_det_hot
                        # max(np.ceil((self.required_connections[self.current_pair] / self.CAP_N[(detector, temp)]) - self.alpha_d[(detector, temp)]) * self.C_det_hot- (1 - (self.required_connections[self.current_pair]/self.CAP_N[d_min_N] - self.alpha_d[d_min_N])%1) * self.C_det_hot, 0)
                if C_curr < C_min_N:
                    C_min_N = C_curr
                    d_min_N = (detector,temp)
                # if costs are equal select the one that leaves the largest fraction alpha over for the next connection
                elif C_curr == C_min_N:
                    alpha_curr = (1-(self.required_connections[self.current_pair]/self.CAP_N[(detector, temp)] - self.alpha_d[(detector, temp)])%1)
                    alpha_min = (1 - (self.required_connections[self.current_pair]/self.CAP_N[d_min_N] - self.alpha_d[d_min_N])%1)
                    if alpha_curr >= alpha_min:
                        C_min_N = C_curr
                        d_min_N = (detector,temp)
        # find the lowest cost & detector for the set CAP_F
        C_min_F = np.infty
        d_min_F = None
        C_min_altered = np.infty
        d_min_altered = None
        for detector, temp in self.CAP_F.keys():
            if self.CAP_F[(detector,temp)] > 0.00001:
                if temp == "c":
                    N = self.get_N(d_m = (detector,temp))
                    C_curr = (self.required_connections[self.current_pair] / self.CAP_F[(detector, temp)]) * self.C_det_cold + self.C_node_cold / N
                        # np.ceil(self.required_connections[self.current_pair] / self.CAP_F[(detector,temp)]) * self.C_det_cold + self.C_node_cold - (1 - (self.required_connections[self.current_pair]/self.CAP_F[d_min_N] - self.alpha_d[d_min_N])%1) * self.C_det_cold
                    C_curr_altered = self.C_node_cold
                    for key in self.CAP_ij:
                        if self.CAP_ij[key][(detector, temp)] < 0.00001:
                            min_det = None
                            min_cost = np.infty
                            for det in self.off_detectors:
                                N_det = self.get_N(d_m = det)
                                if min_det == None:
                                    min_det = det
                                    if det[1] == "c" and self.CAP_F[det] > 0.00001:
                                        min_cost = (self.required_connections[key] * self.C_det_cold / self.CAP_F[det] + self.C_node_cold / N_det)
                                    elif det[1] == "h" and self.CAP_F[det] > 0.00001:
                                        min_cost = (self.required_connections[key] * self.C_det_hot / self.CAP_F[det] + self.C_node_hot / N_det)
                                else:
                                    if det[1] == "c" and self.CAP_F[det] > 0.00001:
                                        min_cost_curr = (self.required_connections[key] * self.C_det_cold / self.CAP_F[det] + self.C_node_cold / N_det)
                                    elif det[1] == "h" and self.CAP_F[det] > 0.00001:
                                        min_cost_curr = (self.required_connections[key] * self.C_det_hot / self.CAP_F[
                                            det] + self.C_node_hot/ N_det)
                                    else:
                                        min_cost_curr = np.infty
                                    if min_cost_curr < min_cost:
                                        min_det = det
                                        min_cost = min_cost_curr
                        else:
                            min_cost = (self.required_connections[key] / self.CAP_F[(detector,temp)]) * self.C_det_cold
                            min_det = None
                            min_cost_new_det = np.infty
                            for det in self.off_detectors:
                                N_det = self.get_N(d_m = det)
                                if min_det == None:
                                    min_det = det
                                    if det[1] == "c" and self.CAP_F[det] > 0.00001:
                                        min_cost_new_det = (self.required_connections[key] * self.C_det_cold / self.CAP_F[
                                            det] + self.C_node_cold / N_det)
                                    elif det[1] == "h" and self.CAP_F[det] > 0.00001:
                                        min_cost_new_det = (self.required_connections[key] * self.C_det_hot / self.CAP_F[
                                            det] + self.C_node_hot/ N_det)
                                else:
                                    if det[1] == "c" and self.CAP_F[det] > 0.00001:
                                        min_cost_curr = (self.required_connections[key] * self.C_det_cold / self.CAP_F[
                                            det] + self.C_node_cold/N_det)
                                    elif det[1] == "h" and self.CAP_F[det] > 0.00001:
                                        min_cost_curr = (self.required_connections[key] * self.C_det_hot / self.CAP_F[
                                            det] + self.C_node_hot/N_det)
                                    else:
                                        min_cost_curr = np.infty
                                    if min_cost_curr < min_cost_new_det:
                                        min_det = det
                                        min_cost_new_det = min_cost_curr
                                if min_cost_new_det < min_cost:
                                    min_cost = min_cost_new_det
                        C_curr_altered += min_cost
                elif temp == "h":
                    N = self.get_N(d_m = (detector,temp))
                    C_curr = (self.required_connections[self.current_pair] / self.CAP_F[(detector, temp)]) * self.C_det_hot + self.C_node_hot / N
                        # np.ceil(self.required_connections[self.current_pair] / self.CAP_F[(detector, temp)]) * self.C_det_hot + self.C_node_hot- (1 - (self.required_connections[self.current_pair]/self.CAP_F[d_min_N] - self.alpha_d[d_min_N])%1) * self.C_det_hot
                    C_curr_altered = self.C_node_hot
                    for key in self.CAP_ij:
                        if self.CAP_ij[key][(detector, temp)] < 0.00001:
                            min_det = None
                            min_cost = np.infty
                            for det in self.off_detectors:
                                N_det = self.get_N(d_m = det)
                                if min_det == None:
                                    min_det = det
                                    if det[1] == "c" and self.CAP_F[det] > 0.00001:
                                        min_cost = (self.required_connections[key] * self.C_det_cold / self.CAP_F[
                                            det] + self.C_node_cold/N_det)
                                    elif det[1] == "h" and self.CAP_F[det] > 0.00001:
                                        min_cost = (self.required_connections[key] * self.C_det_hot / self.CAP_F[
                                            det] + self.C_node_hot/N_det)
                                else:
                                    if det[1] == "c" and self.CAP_F[det] > 0.00001:
                                        min_cost_curr = (self.required_connections[key] * self.C_det_cold / self.CAP_F[
                                            det] + self.C_node_cold/N_det)
                                    elif det[1] == "h" and self.CAP_F[det] > 0.00001:
                                        min_cost_curr = (self.required_connections[key] * self.C_det_hot / self.CAP_F[
                                            det] + self.C_node_hot/N_det)
                                    else:
                                        min_cost_curr = np.infty
                                    if min_cost_curr < min_cost:
                                        min_det = det
                                        min_cost = min_cost_curr
                        else:
                            min_cost = (self.required_connections[key] / self.CAP_F[(detector, temp)]) * self.C_det_hot
                            min_det = None
                            min_cost_new_det = np.infty
                            for det in self.off_detectors:
                                N_det = self.get_N(d_m = det)
                                if min_det == None:
                                    min_det = det
                                    if det[1] == "c" and self.CAP_F[det] > 0.00001:
                                        min_cost_new_det = (
                                                    self.required_connections[key] * self.C_det_cold / self.CAP_F[
                                                det] + self.C_node_cold/N_det)
                                    elif det[1] == "h" and self.CAP_F[det] > 0.00001:
                                        min_cost_new_det = (
                                                    self.required_connections[key] * self.C_det_hot / self.CAP_F[
                                                det] + self.C_node_hot/N_det)
                                else:
                                    if det[1] == "c" and self.CAP_F[det] > 0.00001:
                                        min_cost_curr = (self.required_connections[key] * self.C_det_cold / self.CAP_F[
                                            det] + self.C_node_cold/N_det)
                                    elif det[1] == "h" and self.CAP_F[det] > 0.00001:
                                        min_cost_curr = (self.required_connections[key] * self.C_det_hot / self.CAP_F[
                                            det] + self.C_node_hot/N_det)
                                    else:
                                        min_cost_curr = np.infty
                                    if min_cost_curr < min_cost_new_det:
                                        min_det = det
                                        min_cost_new_det = min_cost_curr
                                if min_cost_new_det < min_cost:
                                    min_cost = min_cost_new_det
                        C_curr_altered += min_cost
                if C_curr <= C_min_F:
                    C_min_F = C_curr
                    d_min_F = (detector, temp)
                elif C_curr == C_min_F:
                    alpha_curr = (1 - (self.required_connections[self.current_pair] / self.CAP_F[(detector, temp)] - self.alpha_d[
                                (detector, temp)]) % 1)
                    alpha_min = (1 - (
                                self.required_connections[self.current_pair] / self.CAP_F[d_min_F] - self.alpha_d[d_min_F]) % 1)
                    if alpha_curr >= alpha_min:
                        C_min_F = C_curr
                        d_min_F = (detector, temp)
                if C_curr_altered <= C_min_altered:
                    C_min_altered = C_curr_altered
                    d_min_altered = (detector,temp)
        if C_min_N >= C_min_F:
            # if we need to turn on a node, we wish to consider the modification of C_node
            if d_min_altered[1] == "c":
                C_min_F = np.ceil(self.required_connections[self.current_pair] / self.CAP_F[d_min_altered])  * self.C_det_cold + self.C_node_cold
            else:
                C_min_F = np.ceil(self.required_connections[self.current_pair] / self.CAP_F[d_min_altered])  * self.C_det_hot + self.C_node_hot
            return C_min_F, d_min_altered, True
        else:
            if d_min_N[1] == "c":
                C_min_N = max(np.ceil(self.required_connections[self.current_pair] / self.CAP_N[d_min_N] - self.alpha_d[d_min_N]) * self.C_det_cold, 0)
            else:
                C_min_N = max(np.ceil(self.required_connections[self.current_pair] / self.CAP_N[d_min_N] - self.alpha_d[d_min_N]) * self.C_det_hot, 0)
            return C_min_N, d_min_N, False

    def update_parameters_new_detector_turned_on(self, d_on):
        self.on_detectors = self.on_detectors.union(set([d_on]))
        self.off_detectors = self.off_detectors - set([d_on])
        self.N_d[d_on] = np.ceil(self.required_connections[self.current_pair] / self.CAP_ij[self.current_pair][d_on])
        self.alpha_d[d_on] = 1-(self.required_connections[self.current_pair] / self.CAP_ij[self.current_pair][d_on])%1
        self.D_used_ij[self.current_pair] = self.D_used_ij[self.current_pair].union(set([d_on]))

    def update_parameters_no_new_detector(self, d_update):
        self.N_d[d_update] = self.N_d[d_update] + max(np.ceil(self.required_connections[self.current_pair] / self.CAP_ij[self.current_pair][d_update] - self.alpha_d[d_update]), 0)
        self.alpha_d[d_update] = 1 - (self.required_connections[self.current_pair] / self.CAP_ij[self.current_pair][d_update] - self.alpha_d[d_update])%1
        self.D_used_ij[self.current_pair] = self.D_used_ij[self.current_pair].union(set([d_update]))

    def run_loop(self):
        # step 5. in the algorithm (stop condition)
        while self.mu < 0:
            print("New pair searching"  + str(self.current_pair))
            # step 1. in the algorithm
            self.split_cap_ij()
            C_min, d_min, new_node_needed = self.lowest_current_cost_detector()
            # step 2. in the algorithm
            if new_node_needed:
                self.update_parameters_new_detector_turned_on(d_on = d_min)
            else:
                self.update_parameters_no_new_detector(d_update=d_min)
            # step 3. in the algorithm
            self.C_tot = self.C_tot + C_min
            self.unsearched_pairs = self.unsearched_pairs - set([self.current_pair])
            self.searched_pairs = self.searched_pairs.union(set([self.current_pair]))
            # step 4. in the algorithm
            if len(self.unsearched_pairs) == 0:
                self.mu = self.mu + 1
                self.unsearched_pairs = copy.deepcopy(self.connection_pairs)
                self.searched_pairs = set()
                print("All pairs searched. Setting new mu term: " + str(self.mu))
            else:
                # step 6. in algorithm.
                current_node = None
                current_capacity = np.infty
                for pair in self.unsearched_pairs:
                    self.CAP_N = {}
                    for key in self.CAP_ij[pair].keys():
                        if key in self.on_detectors.difference(self.D_used_ij[pair]):
                            self.CAP_N[key] = self.CAP_ij[pair][key]
                    self.CAP_N = sorted(self.CAP_N.items(), key=lambda item: item[1], reverse=True)
                    if self.CAP_N[0][1] <= current_capacity:
                        current_node = pair
                        current_capacity = self.CAP_N[0][1]
                self.current_pair = current_node
                print("Finished pair. Starting new pair: " + str(self.current_pair))

    def get_solution(self):
        C_tot = self.C_tot
        detectors_on = copy.deepcopy(self.on_detectors)
        detectors_off = copy.deepcopy(self.off_detectors)
        number_detectors = copy.deepcopy(self.N_d)
        return C_tot, detectors_on, detectors_off, number_detectors




if __name__ == "__main__":
    hot_capacity_dict_multiple_graphs, cold_capacity_dict_multiple_graphs, required_connections_multiple_graphs, distances = import_switched_network_values_multiple_graphs(hot_bob_capacity_file = "mesh_topology_75_hot_capacity_data_2", cold_bob_capacity_file = "mesh_topology_75_cold_capacity_data_2", cmin = 10000)
    model = Heuristic_Model(hot_capacity_dict  = hot_capacity_dict_multiple_graphs[19], cold_capacity_dict = cold_capacity_dict_multiple_graphs[19], required_connections = required_connections_multiple_graphs[19], C_node_cold = 10, C_node_hot= 4, C_det_cold  = 2,
                 C_det_hot = 1, M = 2)
    if model.check_feasible():
        model.run_loop()
        C_tot, detectors_on, detectors_off, number_detectors = model.get_solution()
        print("total cost: "+ str(C_tot))
        for detector, temp in number_detectors.keys():
            print("Number of detectors on Node "  + str(detector) + " at temperature " + temp + ": " + str(number_detectors[(detector,temp)]))