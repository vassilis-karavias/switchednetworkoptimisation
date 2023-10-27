import pandas as pd
import networkx as nx


def import_switched_network_values(hot_bob_capacity_file, cold_bob_capacity_file, cmin):
    hot_bob_capacities = pd.read_csv(hot_bob_capacity_file + ".csv")
    cold_bob_capacities = pd.read_csv(cold_bob_capacity_file + ".csv")

    hot_capacity_dict = {}
    for index, row in hot_bob_capacities.iterrows():
        hot_capacity_dict[(int(row["source"]), int(row["target"]), int(row["detector"]))] = row["capacity"]
    cold_capacity_dict = {}
    for index, row in cold_bob_capacities.iterrows():
        cold_capacity_dict[(int(row["source"]), int(row["target"]), int(row["detector"]))] = row["capacity"]

    required_connections = {}
    for index, row in hot_bob_capacities.iterrows():
        if not required_connections:
            required_connections[(int(row["source"]), int(row["target"]))] = cmin
        elif (int(row["source"]), int(row["target"])) not in required_connections:
            required_connections[(int(row["source"]), int(row["target"]))] = cmin
    for index, row in cold_bob_capacities.iterrows():
        if not required_connections:
            required_connections[(int(row["source"]), int(row["target"]))] = cmin
        elif (int(row["source"]), int(row["target"])) not in required_connections:
            required_connections[(int(row["source"]), int(row["target"]))] = cmin
    return hot_capacity_dict, cold_capacity_dict, required_connections



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


def import_switched_network_values_multiple_graphs_nonuniform_tij(hot_bob_capacity_file, cold_bob_capacity_file, required_conn_file):
    hot_bob_capacities = pd.read_csv(hot_bob_capacity_file + ".csv")
    cold_bob_capacities = pd.read_csv(cold_bob_capacity_file + ".csv")
    connection_caps = pd.read_csv(required_conn_file + ".csv")
    possible_ids = hot_bob_capacities["ID"].unique()
    hot_capacity_dict_multiple_graphs = {}
    cold_capacity_dict_multiple_graphs = {}
    required_connections_multiple_graphs = {}
    distances = {}
    for id in possible_ids:
        hot_bob_capacities_id = hot_bob_capacities[hot_bob_capacities["ID"] == id].drop(["ID"], axis = 1)
        cold_bob_capacities_id = cold_bob_capacities[cold_bob_capacities["ID"] == id].drop(["ID"], axis=1)
        required_conns_id = connection_caps[connection_caps["ID"] == id].drop(["ID"], axis = 1)
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
        for index, row in required_conns_id.iterrows():
            if not required_connections:
                required_connections[(int(row["source"]), int(row["target"]))] = row["key"]
            elif (int(row["source"]), int(row["target"])) not in required_connections:
                required_connections[(int(row["source"]), int(row["target"]))] = row["key"]
        required_connections_multiple_graphs[id] = required_connections
        print("Finished: " + str(id))
    return hot_capacity_dict_multiple_graphs, cold_capacity_dict_multiple_graphs, required_connections_multiple_graphs, distances

def import_graph_structure(node_information, edge_information):
    node_data = pd.read_csv(node_information+ ".csv")
    edge_data = pd.read_csv(edge_information+ ".csv")
    possible_ids = node_data["ID"].unique()
    graphs = {}
    for id in possible_ids:
        node_data_current = node_data[node_data["ID"] == id].drop(["ID"], axis = 1)
        edge_data_current = edge_data[edge_data["ID"] == id].drop(["ID"], axis = 1)
        graph = nx.from_pandas_edgelist(edge_data_current, "source", "target", ["distance"])
        graph = graph.to_undirected()
        graph = graph.to_directed()
        node_attr = node_data_current.set_index("node").to_dict("index")
        nx.set_node_attributes(graph, node_attr)
        graphs[id] = graph
    return graphs