import minimum_length
import capacity_calculation
import os
import csv
from generate_graph import NodeType

def store_capacities(graph, dictionary, store_location, graph_id, size = 100):
    true_distance = graph.get_shortest_distance_of_source_nodes_with_switches()
    # get the total number of detectors in the graph
    vertices = graph.get_vertices()
    # keeps track of the position in the nodeposns array
    n = 0
    # iterate over every vertex
    for vertex in range(len(vertices)):
        # if vertex is a place without Bob and needs to be a Bob then change the node to have a Bob -
        if graph.vertex_type[vertices[vertex]] == NodeType(1).name or graph.vertex_type[vertices[vertex]] == NodeType(2).name:
            n += 1
    # get the shortest distance to any Bob
    distance_array = minimum_length.get_minimum_length_to_each_bob_for_any_source_pair(true_distance, graph.g)
    # get the capacities of for each of the pairs of sources
    capacities, distance_array = capacity_calculation.calculate_capacity_for_n_highest_capacities_efficient_corrected(distance_array, dictionary, n = n)
    capacity_dictionary = []
    for source, target, bob, capacity in capacities:
        capacity_dictionary.append({"ID": graph_id, "source": source, "target": target, "detector": bob, "capacity": capacity, "size": size})
    dictionary_fieldnames = ["ID", "source", "target", "detector", "capacity", "size"]
    if os.path.isfile(store_location + '.csv'):
        with open(store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
            writer.writerows(capacity_dictionary)
    else:
        with open(store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
            writer.writeheader()
            writer.writerows(capacity_dictionary)




def store_capacities_channel_allocation_routing(graph, dictionary, store_location, graph_id, db_switch, size = 100):
    true_distance = graph.get_shortest_distance_of_source_nodes_with_switches_channel_allocation_routing(db_switch)
    # get the total number of detectors in the graph
    vertices = graph.get_vertices()
    # keeps track of the position in the nodeposns array
    n = 0
    # iterate over every vertex
    for vertex in range(len(vertices)):
        # if vertex is a place without Bob and needs to be a Bob then change the node to have a Bob -
        if graph.vertex_type[vertices[vertex]] == NodeType(1).name or graph.vertex_type[vertices[vertex]] == NodeType(2).name:
            n += 1
    # get the shortest distance to any Bob
    distance_array = minimum_length.get_minimum_length_to_each_bob_for_any_source_pair(true_distance, graph.g)
    # get the capacities of for each of the pairs of sources
    capacities, distance_array = capacity_calculation.calculate_capacity_for_n_highest_capacities_efficient_corrected(distance_array, dictionary, n = n)
    capacity_dictionary = []
    for source, target, bob, capacity in capacities:
        capacity_dictionary.append({"ID": graph_id, "source": source, "target": target, "detector": bob, "capacity": capacity, "size": size})
    dictionary_fieldnames = ["ID", "source", "target", "detector", "capacity", "size"]
    if os.path.isfile(store_location + '.csv'):
        with open(store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
            writer.writerows(capacity_dictionary)
    else:
        with open(store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
            writer.writeheader()
            writer.writerows(capacity_dictionary)


def store_capacities_for_hot_cold_bobs(graphs, store_loc_cold, store_location_cold, store_loc_hot, store_location_hot, node_data_store_location, edge_data_store_location, size = 100):
    """

    :param graphs: graphs to use
    :param store_loc_cold: Where the information of capacities vs distance are stored for cold detectors (no .csv for all)
    :param store_location_cold: Where to store information of capacities
    :param store_loc_hot: Where the information of capacities vs distance are stored for hot detectors
    :param store_location_hot: Where to store information of capacities
    :param size: Array of the size of the graphs
    :return:
    """
    dictionary_cold = {}
    with open(store_loc_cold + '.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            dictionary_cold["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
            line_count += 1
        print(f'Processed {line_count} lines.')

    dictionary_hot = {}
    with open(store_loc_hot + '.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            dictionary_hot["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
            line_count += 1
        print(f'Processed {line_count} lines.')
    id = 0
    for i in range(len(graphs)):
        store_capacities(graphs[i], dictionary_cold, store_location_cold, graph_id= id, size = size[i])
        store_capacities(graphs[i], dictionary_hot, store_location_hot, graph_id= id, size = size[i])
        store_position_graph(graphs[i], node_data_store_location, edge_data_store_location, graph_id = id)
        print("Finished graph " + str(i))
        id += 1

def store_capacities_for_hot_cold_bobs_channel_allocation_routing(graphs, store_loc_cold, store_location_cold, store_loc_hot, store_location_hot, node_data_store_location, edge_data_store_location, db_switch, size = 100):
    """

    :param graphs: graphs to use
    :param store_loc_cold: Where the information of capacities vs distance are stored for cold detectors (no .csv for all)
    :param store_location_cold: Where to store information of capacities
    :param store_loc_hot: Where the information of capacities vs distance are stored for hot detectors
    :param store_location_hot: Where to store information of capacities
    :param size: Array of the size of the graphs
    :return:
    """
    dictionary_cold = {}
    with open(store_loc_cold + '.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            dictionary_cold["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
            line_count += 1
        print(f'Processed {line_count} lines.')

    dictionary_hot = {}
    with open(store_loc_hot + '.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            dictionary_hot["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
            line_count += 1
        print(f'Processed {line_count} lines.')
    id = 0
    for i in range(len(graphs)):
        store_capacities_channel_allocation_routing(graphs[i], dictionary_cold, store_location_cold, graph_id= id, size = size[i], db_switch = db_switch)
        store_capacities_channel_allocation_routing(graphs[i], dictionary_hot, store_location_hot, graph_id= id, size = size[i], db_switch = db_switch)
        store_position_graph(graphs[i], node_data_store_location, edge_data_store_location, graph_id = id)
        print("Finished graph " + str(i))
        id += 1


def store_position_graph(network, node_data_store_location, edge_data_store_location, graph_id = 0):
    edges = network.g.get_edges(eprops=[network.lengths_of_connections])
    dictionaries = []
    dictionary_fieldnames = ["ID", "source", "target", "distance"]
    for edge in range(len(edges)):
        source = edges[edge][0] + 1
        target = edges[edge][1] + 1
        distance = edges[edge][2]
        dictionaries.append(
            {"ID": graph_id, "source": source , "target": target, "distance": distance})
    nodes = network.g.get_vertices(vprops =[network.x_coord, network.y_coord])
    dictionary_fieldnames_nodes = ["ID", "node", "xcoord", "ycoord", "type"]
    dict_nodes = []
    for node in range(len(nodes)):
        node_label = nodes[node][0]
        xcoord = nodes[node][1]
        ycoord = nodes[node][2]
        type = network.vertex_type[node_label]
        dict_nodes.append({"ID": graph_id, "node": node_label+1, "xcoord": xcoord, "ycoord": ycoord, "type": type})

    if os.path.isfile(node_data_store_location + '.csv'):
        with open(node_data_store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
            writer.writerows(dict_nodes)
    else:
        with open(node_data_store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
            writer.writeheader()
            writer.writerows(dict_nodes)

    if os.path.isfile(edge_data_store_location + '.csv'):
        with open(edge_data_store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
            writer.writerows(dictionaries)
    else:
        with open(edge_data_store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
            writer.writeheader()
            writer.writerows(dictionaries)