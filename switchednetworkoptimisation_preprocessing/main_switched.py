from capacity_storage import *
from random_graph_generation import generate_random_graph_no_solving, SpecifiedTopologyGraph
from generate_graph import ExistingNetwork
import numpy as np
from enum import Enum, unique
from graph_tool.all import *
import csv

@unique
class Topology(Enum):
    """

    """
    BUS = 0
    RING = 1
    STAR = 2
    MESH = 3
    HUBSPOKE = 4
### parameters to what kind of investigation this is

topology =  Topology(3).name
no_of_bobs = 10
no_of_bob_locations = 10
dbswitch = 0
box_size = 100
# for ring topology
radius = 50
# for star topology
central_node_is_detector = False
# for mesh and hub&spoke
no_of_conns_av = 3.5
mesh_composed_of_only_detectors = False
nodes_for_mesh = 3
mesh_in_centre = True



graphs = []
sizes = []
for n in np.arange(start = 10, stop = 55, step = 5):
    for i in range(10):
        graph_node = SpecifiedTopologyGraph()
        if topology == Topology(0).name:
            graph_node.generate_random_bus_graph(n, no_of_bobs, no_of_bob_locations, dbswitch, box_size)
        elif topology == Topology(1).name:
            graph_node.generate_random_ring_graph(n, no_of_bobs, no_of_bob_locations, radius, dbswitch)
        elif topology == Topology(2).name:
            graph_node.generate_random_star_graph(n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, central_node_is_detector)
        elif topology == Topology(3).name:
            try:
                graph_node.generate_random_mesh_graph(n + no_of_bobs, no_of_bobs, no_of_bobs, dbswitch, box_size, no_of_conns_av)
            except ValueError:
                continue
        elif topology == Topology(4).name:
            if mesh_in_centre:
                try:
                    graph_node.generate_hub_spoke_with_hub_in_centre(n, no_of_bobs, no_of_bobs, dbswitch, box_size, mesh_composed_of_only_detectors, nodes_for_mesh, no_of_conns_av)
                except ValueError:
                    continue
            else:
                try:
                    graph_node.generate_random_hub_spoke_graph(n, no_of_bobs, no_of_bobs, dbswitch, box_size, mesh_composed_of_only_detectors, nodes_for_mesh, no_of_conns_av)
                except ValueError:
                    continue
        print("Finished Graph " + str(no_of_bobs) + "," + str(i))
        graph = graph_node.graph
        graphs.append(graph)
        sizes.append(100)


sizes = [100]
##### Import data for graph create g and use to GeneralNetwork class input...

# edge_list = []
# edge_weights ={}
# pos = {}
# node_types = {}
# with open('test_real_graph_edges_2.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         edge_list.append([int(row["source"])-1, int(row["target"])-1])
#         edge_weights[int(row["source"])-1, int(row["target"])-1] = float(row["weight"])

# with open('real_graph_node_position_data_2.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         pos[int(row["node"])-1] = [float(row["xcoord"]),float(row["ycoord"])]
#         node_types[int(row["node"])-1] = row["type"]


# g = Graph(directed=False)
# g.add_edge_list(edge_list)
# x_coords = g.new_vertex_property(value_type="double")
# y_coords = g.new_vertex_property(value_type="double")  #
# vertex_type = g.new_vertex_property(value_type="object")
# g.vertex_properties["node_type"] = vertex_type
# vertices = g.get_vertices()
# for vertex in vertices:
#     vertex_type[vertices[vertex]] = node_types[vertex]
#     x_coords[vertex] = pos[vertex][0]
#     y_coords[vertex] = pos[vertex][1]
# g.vertex_properties["x_coord"] = x_coords
# g.vertex_properties["y_coord"] = y_coords
# edges = g.get_edges()
# edge_non_rounded = []
# edges_rounded = []
# edges_with_switch = []
# for edge in edges:
#     source_node = edge[0]
#     target_node = edge[1]
#     length_of_connection = edge_weights[source_node, target_node]
#     length_rounded = int(length_of_connection)
#     edges_rounded.append(length_rounded)
#     edge_non_rounded.append(length_of_connection)
#     length_with_switch = length_of_connection + 5 * dbswitch
#     edges_with_switch.append(length_with_switch)
#     # add the length as an edge property
# lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
# lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
# lengths_with_switch = g.new_edge_property(value_type="double", vals=edges_with_switch)

# node_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
#               "11", "12", "13", "14", "15"]
# for i in range(16, 500):
#     node_names.append(str(i))

# graph = ExistingNetwork(g,lengths_of_connections, lengths_rounded, lengths_with_switch, x_coords, y_coords, node_types,label = node_names, dbswitch =1)
# graphs =[graph]
# for parameter sweep on the TF-QKD problem we just need to call this method with multiple store_loc_cold, store_loc_hot
# for switch_loss in np.arange(start = 0.5, stop = 6, step = 0.25):
#     for graph in  graphs:
#         graph.update_db_switch(switch_loss)
#     store_capacities_for_hot_cold_bobs_channel_allocation_routing(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_80_eff",
#                                    store_location_cold= f"heuristic_nodes_mesh_topology_35_cold_capacity_switch_loss_{round(switch_loss,2)}",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_15_eff',
#                                    store_location_hot=f"heuristic_nodes_mesh_topology_35_hot_capacity_switch_loss_{round(switch_loss,2)}",
#                                    node_data_store_location=f"heuristic_nodes_mesh_topology_35_node_positions_switch_loss_{round(switch_loss,2)}",
#                                    edge_data_store_location=f"heuristic_nodes_mesh_topology_35_edge_positions_switch_loss_{round(switch_loss,2)}", size=sizes, db_switch=switch_loss)


# for graph in  graphs:
#     graph.update_db_switch(0)
store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_80_eff",
                                   store_location_cold= f"1_nodes_mesh_topology_35_cold_capacity_no_switch_loss",
                                   store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_15_eff',
                                   store_location_hot=f"1_nodes_mesh_topology_35_hot_capacity_no_switch_loss",
                                   node_data_store_location=f"1_nodes_mesh_topology_35_node_positions_no_switch_loss",
                                   edge_data_store_location=f"1_nodes_mesh_topology_35_edge_positions_no_switch_loss", size=sizes)

# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_80_eff",
#                                    store_location_cold= "real_graph_cold_capacity_2",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_15_eff',
#                                    store_location_hot="real_graph_hot_capacity_2",
#                                    node_data_store_location="real_graph_node_positions_2",
#                                    edge_data_store_location="real_graph_edge_positions_2", size=sizes)
# for graph in  graphs:
#     graph.update_db_switch(0)
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_80_eff",
#                                    store_location_cold= "10_nodes_mesh_topology_35_cold_capacity_no_switch",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_15_eff',
#                                    store_location_hot="10_nodes_mesh_topology_35_hot_capacity_no_switch",
#                                    node_data_store_location="10_nodes_mesh_topology_35_node_positions_no_switch",
#                                    edge_data_store_location="10_nodes_mesh_topology_35_edge_positions_no_switch", size=sizes)

# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_70_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_70_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_15_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_70_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_70_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_70_eff", size=sizes)
#
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_80_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_10_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_10_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_10_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_10_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_10_eff", size=sizes)
#
#
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_80_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_20_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_20_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_20_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_20_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_20_eff", size=sizes)
#
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_80_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_25_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_25_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_25_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_25_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_25_eff", size=sizes)
#
#
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_70_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_70_eff_10_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_10_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_70_eff_10_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_70_eff_10_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_70_eff_10_eff", size=sizes)
#
#
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_70_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_70_eff_20_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_20_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_70_eff_20_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_70_eff_20_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_70_eff_20_eff", size=sizes)
#
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_70_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_70_eff_25_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_25_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_70_eff_25_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_70_eff_25_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_70_eff_25_eff", size=sizes)
#
#
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_90_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_90_eff_10_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_10_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_90_eff_10_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_90_eff_10_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_90_eff_10_eff", size=sizes)
#
#
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_90_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_90_eff_20_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_20_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_90_eff_20_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_90_eff_20_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_90_eff_20_eff", size=sizes)
#
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold="/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_coldbob_90_eff",
#                                    store_location_cold= "8_nodes_mesh_topology_35_cold_capacity_90_eff_25_eff",
#                                    store_loc_hot='/home/vass/anaconda3/envs/gt/sources/switchednetworkoptimisation/rates_hotbob_25_eff',
#                                    store_location_hot="8_nodes_mesh_topology_35_hot_capacity_90_eff_25_eff",
#                                    node_data_store_location="8_nodes_mesh_topology_35_node_positions_90_eff_25_eff",
#                                    edge_data_store_location="8_nodes_mesh_topology_35_edge_positions_90_eff_25_eff", size=sizes)

# graphs = []
# sizes = []
# for n in np.arange(start = 20, stop = 40, step = 1):
#     for i in range(20):
#         graph = generate_random_graph_no_solving(n= n, no_of_bobs = 10, no_of_bob_locations = 10, p = 0.8, no_connected_nodes = 7.5, size = 100)
#         graphs.append(graph)
#         sizes.append(100)
# store_capacities_for_hot_cold_bobs(graphs, store_loc_cold = "/home/vass/anaconda3/envs/gt/sources/rates_coldbob_new", store_location_cold = "capacity_cold_20to40_75connections",
#                                    store_loc_hot = '/home/vass/anaconda3/envs/gt/sources/rates_hotbob_new', store_location_hot = "capacity_hot_20to40_75connections", node_data_store_location = "graphs_node_data_20to40_75connections", edge_data_store_location = "graphs_edge_data_20to40_75connections", size = sizes)
# id = 0
# for graph in graphs:
#     store_position_graph(network = graph, node_data_store_location = "graphs_node_data", edge_data_store_location = "graphs_edge_data", graph_id = id)
#     id += 1
