import networkx as nx
import numpy as np
import pandas as pd
import json
import csv


def construct_skeleton_graph(skeleton_path, with_interpolation,
                             step, leaf_node_removal_depth):
    if skeleton_path.endswith('.json'):
        with open(skeleton_path, 'r') as f:
            skeleton_data = json.load(f)
        if with_interpolation:
            skeleton_nodes = add_nodes_from_catmaidJson_with_interpolation(skeleton_data,step)
        else:
            skeleton_nodes = add_nodes_from_catmaidJson(skeleton_data)
    elif skeleton_path.endswith('.csv'):
        skeleton_data = pd.read_csv(skeleton_path)
        skeleton_data.columns = ['skeleton_id', 'treenode_id',
                                 'parent_treenode_id', 'x', 'y', 'z', 'r']
        if with_interpolation:
            skeleton_nodes = add_nodes_from_catmaidCSV_with_interpolation(skeleton_data,step)
        else:
            skeleton_nodes = add_nodes_from_catmaidCSV(skeleton_data)
    edge_connect_nodes_to_parents(skeleton_nodes)
    return remove_leaf_nodes(skeleton_nodes, leaf_node_removal_depth)


# This method connects every node in the graph to its parent node
# (the node specified by its 'parent_id' attribute) to facilitate
# removing the leaf nodes.
def edge_connect_nodes_to_parents(graph):
    for node in graph.nodes:
        parent = graph.nodes[node]['parent_id']
        if not parent is None:
            graph.add_edge(node, parent)
            # Line below might be unnecessary
            graph[node][parent]['error_type'] = ''
    return graph


# This method removes all the leaf nodes (those with 0 or 1 neighbors) 
# from the graph in order to avoid penalizing the model for small, unimportant 
# misclassifications at the ends of a process.
def remove_leaf_nodes(graph, removal_depth):
    for i in range(removal_depth):
        leaf_nodes = []
        for node in graph.nodes:
            if graph.degree[node] <= 1:
                leaf_nodes.append(node)
        for node in leaf_nodes:
            graph.remove_node(node)
    return graph


def add_nodes(graph, node_z, node_y, node_x, treenode_id, parent_treenode_id,
              sk_id):
    graph.add_nodes_from([treenode_id], skeleton_id=sk_id)
    graph.add_nodes_from([treenode_id], x=node_x)
    graph.add_nodes_from([treenode_id], y=node_y)
    graph.add_nodes_from([treenode_id], z=node_z)
    graph.add_nodes_from([treenode_id], parent_id=parent_treenode_id)
    return graph


def add_nodes_from_catmaidJson(JSONdata):
    skeleton_graph = nx.Graph()
    glia_cells_sk = glia_cells_sk_id_Json(JSONdata)
    for sk_id, sk_dict in JSONdata['skeletons'].items():
        if len(sk_dict['treenodes']) < 2:
            continue
        sk_id = int(sk_id)
        for tr_id, tr_dict in sk_dict['treenodes'].items():
            tr_id = int(tr_id)
            skeleton_graph.add_nodes_from([tr_id], skeleton_id=sk_id)
            skeleton_graph.add_nodes_from([tr_id], x=tr_dict['location'][0])
            skeleton_graph.add_nodes_from([tr_id], y=tr_dict['location'][1])
            skeleton_graph.add_nodes_from([tr_id], z=tr_dict['location'][2])
            skeleton_graph.add_nodes_from([tr_id], parent_id=tr_dict['parent_id'])
    for node in skeleton_graph.nodes:
        if skeleton_graph.nodes[node]['skeleton_id'] in glia_cells_sk:
            skeleton_graph.nodes[node]['cell_type'] = 'glia'
        else:
            skeleton_graph.nodes[node]['cell_type'] = 'neuron'
    return skeleton_graph


def add_nodes_from_catmaidJson_with_interpolation(JSONdata, step):
    skeleton_graph = nx.Graph()
    glia_cells_sk = glia_cells_sk_id_Json(JSONdata)
    id_to_start = int(max(max(list(i['treenodes'].keys()))
                          for i in JSONdata['skeletons'].values()))+1
    for sk_id, sk_dict in JSONdata['skeletons'].items():
        if len(sk_dict['treenodes']) < 2:
            continue
        for tr_id, tr_dict in sk_dict['treenodes'].items():
            (skeleton_graph,
             id_to_start) = interpolation_sections_JSON(skeleton_graph,
                                                        sk_id,
                                                        tr_id,
                                                        sk_dict['treenodes'],
                                                        tr_dict,
                                                        id_to_start,
                                                        step)
    for node in skeleton_graph.nodes:
        if skeleton_graph.nodes[node]['skeleton_id'] in glia_cells_sk:
            skeleton_graph.nodes[node]['cell_type'] = 'glia'
        else:
            skeleton_graph.nodes[node]['cell_type'] = 'neuron'
    return skeleton_graph


def interpolation_sections_JSON(graph, sk_id, tr_id, sk_dict, current_dict,
                                id_to_start, step):
    if current_dict['parent_id'] is None:
        graph = add_nodes(graph,
                          current_dict['location'][2],
                          current_dict['location'][1],
                          current_dict['location'][0],
                          int(tr_id),
                          current_dict['parent_id'],
                          int(sk_id))
        next_id_to_start = id_to_start
    else:
        parent_id = current_dict['parent_id']
        parent_dict = sk_dict[str(parent_id)]
        gap = abs(parent_dict['location'][2] - current_dict['location'][2])
        if gap <= step:
            graph = add_nodes(graph,
                              current_dict['location'][2],
                              current_dict['location'][1],
                              current_dict['location'][0],
                              int(tr_id),
                              current_dict['parent_id'],
                              int(sk_id))
            next_id_to_start = id_to_start
        else:
            graph = add_nodes(graph,
                              current_dict['location'][2],
                              current_dict['location'][1],
                              current_dict['location'][0],
                              int(tr_id),
                              id_to_start,
                              int(sk_id))
            gap_z = step*(parent_dict['location'][2] -
                          current_dict['location'][2])/gap
            gap_y = step*(parent_dict['location'][1] -
                          current_dict['location'][1])/gap
            gap_x = step*(parent_dict['location'][0] -
                          current_dict['location'][0])/gap
            next_node_z = gap_z + current_dict['location'][2]
            next_node_y = gap_y + current_dict['location'][1]
            next_node_x = gap_x + current_dict['location'][0]
            next_sk_id = int(sk_id)
            next_treenode_id = id_to_start
            next_parent_id = id_to_start + 1
            graph = add_nodes(graph, next_node_z, next_node_y, next_node_x,
                              next_treenode_id, next_parent_id, next_sk_id)
            while(next_node_z + gap_z)*gap_z < parent_dict['location'][2]*gap_z:
                next_node_z += gap_z
                next_node_y += gap_y
                next_node_x += gap_x
                next_treenode_id += 1
                next_parent_id += 1
                graph = add_nodes(graph, next_node_z, next_node_y, next_node_x,
                                  next_treenode_id, next_parent_id, next_sk_id)
            graph = add_nodes(graph, next_node_z+gap_z, next_node_y+gap_y,
                              next_node_x+gap_x, next_treenode_id+1, parent_id,
                              next_sk_id)
            next_id_to_start = next_treenode_id+2
    return graph, next_id_to_start

	
def glia_cells_sk_id_Json(JSONdata):
    glia_cell_sk_id = set() #store the sk_id which are glia cells
    parent_dict = {} # store the parent_treenode_id and their child 
    for sk_id, sk_dict in JSONdata['skeletons'].items():
        parent_dict.clear()
        for tr_id, tr_dict in sk_dict['treenodes'].items():
            if tr_dict['parent_id'] is None:
                continue
            elif tr_dict['parent_id'] not in parent_dict:
                parent_dict[tr_dict['parent_id']] = set()
                parent_dict[tr_dict['parent_id']].add(int(tr_id)) 
            else:
                parent_dict[tr_dict['parent_id']].add(int(tr_id))
        for child_set in parent_dict.values():
            if len(child_set) > 5:
                child_count = 0
                for child in child_set: 
                    if child not in parent_dict:
                        child_count += 1
                if child_count > 5:
                    glia_cell_sk_id.add(int(sk_id))  
    return glia_cell_sk_id 
	

##################### CSV Versions ##################### 
def add_nodes_from_catmaidCSV(CSVdata, ignore_glia=True):
    skeleton_graph = nx.Graph()
    # print(data.columns.values)
    glia_cells_sk = glia_cells_sk_id_CSV(CSVdata)
    for i, nrow in CSVdata.iterrows():
        if ignore_glia and current_row['skeleton_id'] in glia_cells_sk:
            continue
        else: 
            skeleton_graph.add_nodes_from([nrow['treenode_id']], skeleton_id=nrow['skeleton_id'])
            skeleton_graph.add_nodes_from([nrow['treenode_id']], x=nrow['x'])
            skeleton_graph.add_nodes_from([nrow['treenode_id']], y=nrow['y'])
            skeleton_graph.add_nodes_from([nrow['treenode_id']], z=nrow['z'])
            skeleton_graph.add_nodes_from([nrow['treenode_id']], parent_id=nrow['parent_treenode_id'])
    return skeleton_graph


def add_nodes_from_catmaidCSV_with_interpolation(CSVdata, step, ignore_glia = True):
    graph = nx.Graph()
    glia_cells_sk = glia_cells_sk_id_CSV(CSVdata)
    id_to_start = max(CSVdata['treenode_id'])+1
    for _, current_row in CSVdata.iterrows():
        if ignore_glia and current_row['skeleton_id'] in glia_cells_sk:
            continue
        else:
            graph, id_to_start = interpolation_sections_CSV(graph,
                                                            CSVdata,
                                                        current_row,
                                                        id_to_start,
                            step)
    return graph


def interpolation_sections_CSV(graph, CSVdata, current_row, id_to_start,
                               step):
    if np.isnan(current_row['parent_treenode_id']):
        graph = add_nodes(graph, current_row['z'], current_row['y'],
                          current_row['x'], current_row['treenode_id'],
                          current_row['parent_treenode_id'],
                          current_row['skeleton_id'])
        next_id_to_start = id_to_start
    else:
        parent_id = current_row['parent_treenode_id']
        parent_row = CSVdata[CSVdata['treenode_id'] == parent_id].squeeze()
        skeleton_id = current_row['skeleton_id']
        gap = abs(parent_row['z'] - current_row['z'])
        if gap <= step:
            graph = add_nodes(graph, current_row['z'], current_row['y'],
                              current_row['x'], current_row['treenode_id'],
                              current_row['parent_treenode_id'],
                              current_row['skeleton_id'])
            next_id_to_start = id_to_start
        else:
            graph = add_nodes(graph, current_row['z'], current_row['y'],
                              current_row['x'], current_row['treenode_id'],
                              id_to_start, current_row['skeleton_id'])
            gap_z = step*(parent_row['z'] - current_row['z'])/gap
            gap_y = step*(parent_row['y'] - current_row['y'])/gap
            gap_x = step*(parent_row['x'] - current_row['x'])/gap
            next_node_z = gap_z + current_row['z']
            next_node_y = gap_y + current_row['y']
            next_node_x = gap_x + current_row['x']
            next_sk_id = skeleton_id
            next_treenode_id = id_to_start
            next_parent_id = id_to_start + 1
            graph = add_nodes(graph, next_node_z, next_node_y, next_node_x,
                              next_treenode_id, next_parent_id, next_sk_id)
            while (next_node_z + gap_z)*gap_z < parent_row['z']*gap_z:
                next_node_z += gap_z
                next_node_y += gap_y
                next_node_x += gap_x
                next_treenode_id += 1
                next_parent_id += 1
                graph = add_nodes(graph, next_node_z, next_node_y, next_node_x,
                                  next_treenode_id, next_parent_id, next_sk_id)
            graph = add_nodes(graph, next_node_z+gap_z, next_node_y+gap_y,
                              next_node_x+gap_x, next_treenode_id+1, parent_id,
                              next_sk_id)
            next_id_to_start = next_treenode_id+2
    return graph, next_id_to_start


def glia_cells_sk_id_CSV(CSVdata):
    glia_cell_sk_id = set() #store the sk_id which are glia cells
    parent_dict = {} # store the parent_treenode_id and their child 
    sk_id = -1 
    for i, current_row in CSVdata.iterrows():
        if current_row['skeleton_id'] != sk_id: #find the end of this skeleton with sk_id
            for child_set in parent_dict.values():
                if len(child_set) > 5:
                    child_count = 0
                    for child in child_set: 
                        if child not in parent_dict:
                            child_count += 1
                    if child_count > 5:
                        glia_cell_sk_id.add(sk_id)
            sk_id = current_row['skeleton_id']
            parent_dict.clear()

        if np.isnan(current_row['parent_treenode_id']): 
            continue
        elif current_row['parent_treenode_id'] not in parent_dict:
            parent_dict[current_row['parent_treenode_id']] = set()
            parent_dict[current_row['parent_treenode_id']].add(current_row['treenode_id'])
        else:  
            parent_dict[current_row['parent_treenode_id']].add(current_row['treenode_id'])
    return glia_cell_sk_id