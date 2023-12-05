# switchednetworkoptimisation
Repository for TF-QKD optimisation


# Requirements
numpy: 1.20.1+  
graph-tool: 2.37+  
pandas: 1.5.2+  
scipy: 1.7.3+  
cplex: V20.1  
matplotlib: 3.6.2+  
networkx: 2.8.8+  
scikit-learn: 1.1.3+  
dill: 0.3.6+  


# How to use:
## Preprocessing
To carry out the optimisations, it is first necessary to have graph files in the appropriate form for the model. This requires 4 csv files: a hot capacity file, a cold capacity file, a node position file and an edge position file. The files can either be provided as is to the model or you can use the methods provided to generate these files for random graphs. To generate random graphs run main_switched.py under the switchednetworkoptimisation_preprocessing folder. You can specify which topology the graphs should have by changing the topology parameter. You can specify how many detector sites there are by changing the no_of_bob_locations parameter. Switch losses can be added by varying the dbswitch parameter and the box in which the network works in can be varied in size by changing the box_size parameter (in km). Once the parameters have been specified it will generate graph between 10 and 50 nodes with 10 graphs of each size. To store the appropriate files needed use the method:  
**store_capacities_for_hot_cold_bobs(graphs, store_loc_cold, store_location_cold, store_loc_hot, store_location_hot, node_data_store_location, edge_data_store_location, size)**  
graphs is an array of required graphs, store_loc_cold is the position of the dictionary that contains a file of distance to capacity for the cold detectors store_location_cold is the file name where you want to store the cold capacity file needed for the optimisation. Similarly for the hot parameters. node_data_store_location is the location where you want to store the node position file and edge_data_store_location is the location where you want to store the edge position file. size is an array which contains the box_size parameter of each graph. For the dictionaries that contain the distance to capacity mappings, we provide files **"rate_coldbob_80_eff.csv", "rate_hotbob_15_eff.csv"** etc. Alternatively, you can provide your own files for these.  
The 4 files will have the following format:  
hot capacity file, cold capacity file: ID, source, target, detector, capacity, size  
node position file: ID, node, xcoord, ycoord, type  
edge position file: ID, source, target, distance  
The ID labels the graph the line belongs to and means multiple graphs can be used in a single file. source and target are the node labels for the user node connections, detector is the node label for the detector of the connection, capacity is the capacity of the connection and size is the overall box_size of the graph. node is the label for the current node, xcoord and ycoord are the coordinates of the node (in km) in the physical network and type labels whether the node is a user node or a detector node in the network. Finally, distance is the distance in km of the edge in the graph.  
## Optimisations
### No Switching
With these files you can now run the optimisation models. These are provided in the switchednetworkoptimisation_optimisation directory. The no-switching model can be ran by calling the  
**sol_dict, prob = optimisation_no_switching.initial_optimisation(hot_key_dict, cold_key_dict, required_connections, cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, time_limit = 1e7, early_stop = 0.003)**  
method. To obtain the hot_key_dict, cold_key_dict and required_connections the method:  
**hot_capacity_dict, cold_capacity_dict, required_connections, distances = switched_network_utils.import_switched_network_values_multiple_graphs(hot_bob_capacity_file, cold_bob_capacity_file, cmin)**  
requiring the hot capacity and cold capacity files obtained in the preprocessing method. cmin is the capacity needed in bits/s for each of tthe connections. A non uniform cmin can be imported using the method:  
**switched_network_utils.import_switched_network_values_multiple_graphs_nonuniform_tij(hot_bob_capacity_file, cold_bob_capacity_file, required_conn_file)**  
which requires an additional csv file (required_conn_file) with format: ID, source, target, capacity. This must be specified by the user and there is currently no method to generate these randomly. cost_det_h is the hot detector cost, cost_det_c is the cold detector cost, cost_on_h(c) is the cost of turning on the hot(cold) detector nodes, N is the maximum number of detectors on a node, M is the robustness parameter. time_limit and early_stop allow you to specify when you want the optimisation to stop. The results are obtained in a sol_dict solution dictionary, which is a dictionary of all the variable names and the values. prob provides the Cplex class of the problem. The objective value can be obtained by:   
**prob.solution.get_objective_value()**  
The physical graph structure can be imported using:   
**switched_network_utils.import_graph_structure(node_information, edge_information)**  
where node_information is the node position file and edge_information is the edge position file.  
### Switching to Any Number of Users
There are 2 switched models of interest. The model which allows any number of users to be shared between devices can be ran using:  
**sol_dict, prob, time_to_solve = optimisation_switching_model.initial_optimisation_switching_fixed_switching_ratio_no_delta(
                        hot_key_dict,
                        cold_key_dict,
                        required_connections,
                        cost_det_h, cost_det_c,
                        cost_on_h, cost_on_c, N, M,
                        fract_switch,
                        time_limit, early_stop)**  
frac_switch specifies the switching calibration time. The files used should now contain the additional switch loss which can be added into the preprocessing.  
### Limited Switching 
The model which limits the number of devices per detector to ω is  
**sol_dict, prob, time_to_solve = optimisation_limited_device_switching.initial_optimisation_switching_limited_number_of_switching(hot_key_dict,
                            cold_key_dict,
                            required_connections,
                            cost_det_h, cost_det_c,
                            cost_on_h, cost_on_c, N, M,
                            omega,
                            fract_switch,
                            time_limit, early_stop)**  
### Methods for in Depth Analysis
Methods to do in depth analysis are provided in optimisation_switching_model and optimisation_limited_device_switching. For example to generate the graph for limited device switching with varying cmin the method:  
**optimisation_limited_device_switching.plot_solution_for_varying_capacity_with_omega(hot_bob_capacity_file, cold_bob_capacity_file, hot_bob_capacity_file_no_switch, cold_bob_capacity_file_no_switch, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, f_switch, data_storage_location_keep_each_loop, data_storage_location_keep_each_loop_no_switch)**  
can be used. data_storage_location_keep_each_loop is an optional file that specifies a location to store the data as the module is solving the necessary problem to ensure data is not lost for the switching terms. the no_switch version of this is the same but for the no-switching terms.  
In optimiation_switching_model, we provide the methods  
*find_critical_ratio_cooled_uncooled(hot_bob_capacity_file, cold_bob_capacity_file, graph_edge_data_file, graph_node_data_file, N, M,f_switch, cost_det_h, cost_on_h, cmin, data_storage_location_keep_each_loop)*  
*time_taken_with_increasing_number_of_nodes(hot_bob_capacity_file, cold_bob_capacity_file,  graph_edge_data_file, graph_node_data_file, cost_det_h, cost_det_c, cost_on_h,
                                      cost_on_c, N, M, frac_switch, cmin)*  
*switch_loss_cost_comparison(cost_det_h, cost_det_c, cost_on_h,
                                      cost_on_c, N, M, frac_switch, cmin=5000, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_no_switch = None)*  
Note: The file names for this are fixed to hot_bob_capacity_file=f"11_nodes_mesh_topology_35_hot_capacity_switch_loss_{round(switch_loss,2)}", cold_bob_capacity_file=f"11_nodes_mesh_topology_35_cold_capacity_switch_loss_{round(switch_loss,2)}", hot_bob_capacity_file_no_switch=f"11_nodes_mesh_topology_35_hot_capacity_switch_loss_no_switch",
        cold_bob_capacity_file_no_switch=f"11_nodes_mesh_topology_35_cold_capacity_switch_loss_no_switch" however this can be changed as required.  

*compare_different_detector_parameter(cost_det_h, cost_det_c, cost_on_h, cost_on_c, N, M, frac_switch, cmin)*
Note: The file names for this are fixed to hot_bob_capacity_file=f"8_nodes_mesh_topology_35_hot_capacity_{eff_cold}_eff_{eff_hot}_eff", cold_bob_capacity_file=f"8_nodes_mesh_topology_35_cold_capacity_{eff_cold}_eff_{eff_hot}_eff" however this can be changed as required.    
*f_switch_parameter_sweep(hot_bob_capacity_file, cold_bob_capacity_file, hot_bob_capacity_file_no_switching, cold_bob_capacity_file_no_switching, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, cmin, data_storage_location_keep_each_loop, data_storage_location_keep_each_loop_no_switch)*  
*cmin_parameter_sweep(hot_bob_capacity_file, cold_bob_capacity_file, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, f_switch, data_storage_location_keep_each_loop)*  
*cmin_parameter_sweep_with_no_switching_comaprison(hot_bob_capacity_file, cold_bob_capacity_file, hot_bob_capacity_no_switching, cold_bob_capacity_no_switching, N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, f_switch, data_storage_location_keep_each_loop, data_storage_location_no_switching)*  
*heuristic_comparison_analysis(hot_bob_capacity_file, cold_bob_capacity_file,N, M,  cost_det_h, cost_det_c, cost_on_h, cost_on_c, cmin, f_switch, data_storage_location_keep_each_loop)*  
### Heuristic
The last method allows for a comparison with the heuristic. This can also be accessed on its own using the Relaxed_heuristic.py file. In order to run this   
**model = LP_relaxation.LP_Switched_fixed_switching_time_relaxation(name, hot_key_dict,
                                                            cold_key_dict,
                                                            Lambda,
                                                            required_connections)**  
sets up the initial problem. name is a placeholder name for the optimisation iteration.  
**heuristic = Relaxed_heuristic.Relaxed_Heuristic(c_det_hot, c_det_cold, c_cold_on, c_hot_on, Lambda, f_switch, M)**  
sets up the heuristic parameters.  
**model_best = heuristic.full_recursion(initial_model= model)**
carries out the heuristic recursion to obtain the best model.
                                                            

