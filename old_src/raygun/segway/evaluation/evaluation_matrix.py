import math
from daisy import Coordinate
import networkx as nx
import copy
import numpy as np
from itertools import combinations, product


def find_merge_errors(graph, z_weight_multiplier, ignore_glia, assume_minimal_merge):
    seg_dict = {}
    for treenode_id, attr in graph.nodes(data=True):
        seg_label = attr['seg_label']
        try:
            seg_dict[seg_label].add(treenode_id)
        except KeyError:
            seg_dict[seg_label] = {treenode_id}

    merge_errors, glia_merges = set(), set()
    for seg_label, nodes in seg_dict.items():
        seg_graph = build_segment_label_subgraph(nodes, graph)
        skel_clusters = list(nx.connected_components(seg_graph))
        if len(skel_clusters) <= 1:
            continue
        potential_merge_sites = []
        for skeleton_1, skeleton_2 in combinations(skel_clusters, 2):
            shortest_connection = get_closest_node_pair_between_two_skeletons(
                                  skeleton_1, skeleton_2, z_weight_multiplier, graph)
            potential_merge_sites.append(shortest_connection)
        
        merge_sites = []
        if assume_minimal_merge:
            merge_sites = list(nx.k_edge_augmentation(seg_graph, k=1,
                                         avail=potential_merge_sites, weight='distance'))
        else:
            merge_sites = [(error_site[0], error_site[1]) for error_site in potential_merge_sites]
        
        for error in merge_sites:
            glial_pair = all([graph.nodes[node]['cell_type'] == 'glia' for node in error])
            if not ignore_glia or not glial_pair:
                merge_errors.add(error)
            if glial_pair:
                glia_merges.add(error)
    return merge_errors, glia_merges


def build_segment_label_subgraph(segment_nodes, graph):
    subgraph = graph.subgraph(segment_nodes)
    skeleton_clusters = nx.connected_components(subgraph)
    seg_graph = nx.Graph()
    seg_graph.add_nodes_from(subgraph.nodes)
    seg_graph.add_edges_from(subgraph.edges)
    for skeleton_1, skeleton_2 in combinations(skeleton_clusters, 2):
        try:
            node_1 = skeleton_1.pop()
            node_2 = skeleton_2.pop()
            if graph.nodes[node_1]['skeleton_id'] == graph.nodes[node_2]['skeleton_id']:
                seg_graph.add_edge(node_1, node_2)
        except KeyError:
            pass
    return seg_graph


# Returns the closest pair of nodes on 2 skeletons
def get_closest_node_pair_between_two_skeletons(skel1, skel2, z_weight_multiplier, graph):
    multiplier = (z_weight_multiplier, 1, 1)
    shortest_len = math.inf
    for node1, node2 in product(skel1, skel2):
        coord1, coord2 = graph.nodes[node1]['zyx_coord'], graph.nodes[node2]['zyx_coord']
        distance = math.sqrt(sum([(a-b)**2 for a, b in zip(map(lambda c,d: c*d, coord1, multiplier),
                                                           map(lambda c,d: c*d, coord2, multiplier))]))
        if distance < shortest_len:
            shortest_len = distance
            edge_attributes = {'distance': shortest_len}
            closest_pair = (node1, node2, edge_attributes)
    return closest_pair


# A split error occurs when a pair of adjacent nodes receive different predicted segmentation IDs
# despite belong to the same neuron. 
def find_split_errors(graph, ignore_glia, max_break_size):
    for edge in graph.edges:
        node1 = edge[0]
        node2 = edge[1]
        if ignore_glia and graph.nodes[node1]['cell_type'] == 'glia':
            continue
        elif graph.nodes[node1]['seg_label'] != graph.nodes[node2]['seg_label']:
            breaks = find_local_breaking_errors(edge, graph, max_break_size)
            if len(breaks):
                for break_edge in breaks:
                    graph.edges[break_edge]['error_type'] = 'breaking'
            else:
                graph.edges[edge]['error_type'] = 'split'
    split_edges = {edge for edge in graph.edges if graph.edges[edge]['error_type'] == 'split'}
    breaking_edges = {edge for edge in graph.edges if graph.edges[edge]['error_type'] == 'breaking'}
    return split_edges, breaking_edges


# A breaking error is a split error in which a mislabeled node or group of mislabeled nodes does not break the continuity of the skeleton it
# belongs to. For example, ...-X-X-X-Y-X-X would be an example of a breaking error of size 1, whereas ...-X-X-X-Y1-Y2-X-X would be an
# example of a breaking error of size 2. Note that Y1 and Y2 need not have the same label, so long as their label or labels differ from
# that of X. If Y1 and Y2 have different ids, Y1-Y2 will also be labelled a breaking error to prevent it from being counted as a split error.
# Breaking errors typically occur when a large organelle is labeled as background, preventing correct agglomeration at that location.
def find_local_breaking_errors(edge, graph, max_size):
    if graph.edges[edge]['error_type'] == 'breaking':
        return {edge}
    node1, node2 = edge[0], edge[1]
    for i in range(2):
        # Begins by constructing the subgraph constituting the max allowable break in the skeleton continuity
        subgraph = {node2}
        for i in range(max_size - 1):
            hits = set()
            for node in subgraph:
                for neighbor in graph.neighbors(node):
                    if graph.nodes[neighbor]['seg_label'] != graph.nodes[node1]['seg_label']:
                        hits.add(neighbor)
            subgraph = subgraph.union(hits)
        # Retrieves all the nodes at the periphery of the subgraph and checks that they all have the correct ID
        borders = set()
        for node in subgraph:
            for neighbor in graph.neighbors(node):
                if neighbor not in subgraph:
                    borders.add(neighbor)
        breaking_error = len(borders) and all(graph.nodes[node]['seg_label'] ==
                                                graph.nodes[node1]['seg_label'] for node in borders)

        # If the error is a breaking error, all the other edges at the periphery of the subgraph (and the adjacent nodes
        # with differing segmentation ID prediction within the subgraph) will be labelled breaking errors)
        if breaking_error == True:
            breaks = set()
            for node in borders:
                for neighbor in graph.neighbors(node):
                    if neighbor in subgraph:
                        breaks.add((node, neighbor))
            for node in subgraph:
                for neighbor in set(graph.neighbors(node)) - borders:
                    if graph.nodes[node]['seg_label'] != graph.nodes[neighbor]['seg_label']:
                        breaks.add((node, neighbor))
            return breaks
        else:
            # if a breaking error is not detected on the first pass, the process is repeated with node1 and node2 reversed. 
            # For example, in ...-X-X-X-Y-X-X-.., if Y was node1 and X node2, the breaking error would not be found until pass 2.
            node1, node2 = node2, node1
    return set()



# 2. rand and voi
# i : gt(skeleton) , j : prediction(segmentation)
# new rand = 1-rand, within the range 0-1, lower is better
# voi could be higher than 1, lower is better
# transform the script below to python code
# https://github.com/funkelab/funlib.evaluate/blob/master/funlib/evaluate/impl/rand_voi.hpp

# Make sensitive to glia
def rand_voi_split_merge(graph, return_cluster_scores=False):
    p_ij = {}
    p_i = {}
    p_j = {}
    total = 0
    for treenode_id, attr in graph.nodes(data=True):
        if attr['seg_label'] == -1:
            continue
        sk_id = attr['skeleton_id']
        seg_id = attr['seg_label']
        total += 1
        if sk_id not in p_i:
            p_i[sk_id] = 1
        else:
            p_i[sk_id] += 1

        if seg_id not in p_j:
            p_j[seg_id] = 1
        else:
            p_j[seg_id] += 1

        if sk_id not in p_ij:
            p_ij[sk_id] = {}
            p_ij[sk_id][seg_id] = 1
        elif seg_id not in p_ij[sk_id]:
            p_ij[sk_id][seg_id] = 1
        else:
            p_ij[sk_id][seg_id] += 1
    # sum of squares in p_ij
    sum_p_ij = 0
    for i_dict in p_ij.values():
        for freq_label in i_dict.values():
            sum_p_ij += freq_label * freq_label
    # sum of squres in p_i
    sum_p_i = 0
    for freq_label in p_i.values():
        sum_p_i += freq_label * freq_label
    # sum of squres in p_j
    sum_p_j = 0
    for freq_label in p_j.values():
        sum_p_j += freq_label * freq_label
    # we have everything we need for RAND, normalize histograms for VOI
    for sk_id, i_dict in p_ij.items():
        for seg_id in i_dict:
            p_ij[sk_id][seg_id] /= total
    for sk_id in p_i:
        p_i[sk_id] /= total
    for seg_id in p_j:
        p_j[seg_id] /= total
    # compute entropies
    voi_split_i = {}
    voi_merge_j = {}
    if return_cluster_scores:
        for sk_id, prob in p_i.items():
            voi_split_i[sk_id] = prob * math.log2(prob)
        for seg_id, freq_label in p_j.items():
            voi_merge_j[seg_id] = prob * math.log2(prob)
    # H(a,b)
    H_ab = 0
    for sk_id, i_dict in p_ij.items():
        for seg_id, prob in i_dict.items():
            H_ab -= prob * math.log2(prob)
            if return_cluster_scores:
                voi_split_i[sk_id] -= prob * math.log2(prob)
                voi_merge_j[seg_id] -= prob * math.log2(prob)
    # H(a)
    H_a = 0
    for prob in p_i.values():
        H_a -= prob * math.log2(prob)
    # H(b)
    H_b = 0
    for prob in p_j.values():
        H_b -= prob * math.log2(prob)
    rand_split = 1-sum_p_ij/sum_p_i
    rand_merge = 1-sum_p_ij/sum_p_j
    # H(b|a)
    voi_split = H_ab - H_a
    # H(a|b)
    voi_merge = H_ab - H_b

    if return_cluster_scores:
        return (rand_split, rand_merge, voi_split, voi_merge), (voi_split_i, voi_merge_j)
    else:
        return (rand_split, rand_merge, voi_split, voi_merge)

