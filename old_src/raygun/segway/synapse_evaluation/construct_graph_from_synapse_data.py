import utility as util
import json
import time
from os import path
from itertools import product
import daisy
from daisy import Coordinate
import numpy as np
import networkx as nx
from scipy import ndimage
from skimage import measure
import os


# This method uses the data in the provided catmaid skeleton json
# to construct a directed graph in which nodes represent pre and postsynaptic sites.
def load_synapses_from_catmaid_json(json_path):
    with open(json_path, 'r') as f:
        catmaid_data = json.load(f)
    conn_dict = {}
    syn_graph = nx.DiGraph()
    for sk_id, sk_dict in catmaid_data['skeletons'].items():
        for conn, attr in sk_dict['connectors'].items():
            if conn not in conn_dict:
                conn_dict[conn] = {}
                zyx_coord = (int(attr['location'][2]),
                             int(attr['location'][1]),
                             int(attr['location'][0]))
                conn_dict[conn]['zyx_coord'] = zyx_coord
                conn_dict[conn]['presyn_sites'] = set()
                conn_dict[conn]['postsyn_sites'] = set()   
            for presyn in attr['presynaptic_to']:
                (x, y, z) = sk_dict['treenodes'][str(presyn)]['location']
                syn_graph.add_node(presyn,
                                   zyx_coord=(int(z), int(y), int(x)),
                                   sk_id=sk_id)
                conn_dict[conn]['presyn_sites'].add(presyn)
            for postsyn in attr['postsynaptic_to']:
                (x, y, z) = sk_dict['treenodes'][str(postsyn)]['location']
                syn_graph.add_node(postsyn,
                                   zyx_coord=(int(z), int(y), int(x)),
                                   sk_id=sk_id)
                conn_dict[conn]['postsyn_sites'].add(postsyn)
    for conn, attr in conn_dict.items():
        for presyn, postsyn in product(attr['presyn_sites'],
                                       attr['postsyn_sites']):
            syn_graph.add_edge(presyn, postsyn,
                               conn_id=conn,
                               conn_zyx_coord=attr['zyx_coord'])
    return syn_graph


def construct_prediction_graph(min_inference_value, model_data,
                               segmentation_data, extraction_config,
                               # voxel_size,
                               configs
                               ):
    inf_graph_json = model_data['inf_graph_json'].format(min_inference_value)

    if os.path.exists(inf_graph_json) and not extraction_config['force_rebuild_db']:
        pred_graph = util.json_to_syn_graph(inf_graph_json)
    else:

        zarr_f = model_data["mask"]
        zarr_full_path = zarr_f['zarr_path'] + '/' + zarr_f['dataset']
        if not os.path.exists(zarr_full_path):
            print(zarr_f)
            raise RuntimeError("ZARR path %s does not exist!" % zarr_full_path)
        roi = daisy.open_ds(zarr_f['zarr_path'], zarr_f['dataset']).roi
        roi_offset = roi.get_offset()
        roi_shape = roi.get_shape()

        pred_graph = extract_mask_sites(**model_data["mask"],
                                                min_inference_value=min_inference_value,
                                                materialize=extraction_config['materialize'],
                                                # partner_type="mask",
                                                )
        pred_graph.graph = {'segmentation': segmentation_data,
                            'synapse_data': {key:model_data[key] for
                                             key in ["mask", 'vector']},
                            'min_inference_value': min_inference_value,
                            # 'voxel_size': voxel_size,
                            'roi_offset': roi_offset,
                            'roi_shape': roi_shape,
                            }

        if configs.mode == "edge_accuracy":
            pred_graph = extract_partner_sites(pred_graph,
                                                   **model_data['vector'],
                                                   # voxel_size=voxel_size,
                                                   materialize=extraction_config['materialize'])

        pred_graph = remove_out_of_roi_synapses(pred_graph, roi)

        if configs.mode == "edge_accuracy":
            if configs.have_segmentation:
                pred_graph = add_segmentation_labels(pred_graph,
                                                     **segmentation_data,
                                                     materialize=extraction_config['materialize'])

                if extraction_config['remove_intraneuron_synapses']:
                    pred_graph = remove_intraneuron_synapses(pred_graph)

    return pred_graph


# This method extracts the connected components from the inference
# array produced by the network. Each connected component receives
# various scores, each of which is positively correlated with its
# probability of being an actual synapse.
# TODO: Maybe consider using mask instead of regionprops
def extract_mask_sites(zarr_path, dataset,
                               min_inference_value,
                               materialize):
    print("Extracting detection sites from {}".format(dataset),
          "at min inference value of {}".format(min_inference_value))
    start_time = time.time()
    prediction_ds = daisy.open_ds(zarr_path, dataset)
    roi = prediction_ds.roi
    voxel_size = prediction_ds.voxel_size
    prediction_ds = prediction_ds[roi]
    if materialize:
        print("Materializing")
        prediction_ds.materialize()
        print("Array materialized")
    inference_array = prediction_ds.to_ndarray()
    labels, _ = ndimage.label(inference_array > min_inference_value)
    extracted_syns = measure.regionprops(labels, inference_array)
    print("Extracted %d synaptic partner" % len(extracted_syns))

    syn_graph = nx.DiGraph()
    for i, syn in enumerate(extracted_syns):
        syn_id = i + 1
        centroid_index = tuple(int(index) for index in syn.centroid)
        zyx_coord = util.np_index_to_daisy_zyx(centroid_index,
                                               voxel_size,
                                               roi.get_offset())
        pixel_coord = util.np_index_to_pixel_xyz(zyx_coord)
        # print(pixel_coord)
        syn_graph.add_node(syn_id,
                           zyx_coord=zyx_coord,
                           pixel_coord=pixel_coord,
                           max=float(syn.max_intensity)/255,
                           area=int(syn.area),
                           mean=float(syn.mean_intensity)/255,
                           sum=int(syn.area * float(syn.mean_intensity)/255))
        # zyx = syn_graph.nodes[syn_id]['zyx_coord']

    print("Extraction took {} seconds".format(time.time()- start_time))
    return syn_graph


def extract_partner_sites(pred_graph, zarr_path,
                              dataset,
                              # voxel_size,
                              materialize):
    # TODO: this function assumes postsyn
    print("Extracting vector predictions from {}".format(dataset))
    start_time = time.time()
    prediction_ds = daisy.open_ds(zarr_path, dataset)
    voxel_size = prediction_ds.voxel_size
    prediction_ds = prediction_ds[prediction_ds.roi]
    # network_constant_z_vector = True
    network_vector_mult = [4, 4, 4]
    materialize = True
    if materialize:
        print("Materializing")
        prediction_ds.materialize()
        print("Array materialized")
    postsyns = list(pred_graph.nodes(data='zyx_coord'))
    for i, (postsyn_id, postsyn_zyx) in enumerate(postsyns):
        presyn_id = postsyn_id * -1
        vector = prediction_ds[postsyn_zyx]
        # if network_constant_z_vector:
        #     vector[0] = 0
        print(vector)
        vector_nm = Coordinate(vector)*Coordinate(network_vector_mult)
        # vector_zyx = vector_nm / Coordinate(voxel_size)
        vector_zyx = vector_nm
        presyn_zyx = vector_zyx + Coordinate(postsyn_zyx)
        pixel_coord = util.np_index_to_pixel_xyz(presyn_zyx)
        pred_graph.add_node(presyn_id,
                            zyx_coord=presyn_zyx,
                            pixel_coord=pixel_coord)
        pred_graph.add_edge(presyn_id, postsyn_id)

        # print(pred_graph.nodes[postsyn_id])
        postsyn_xyz = pred_graph.nodes[postsyn_id]["pixel_coord"]
        # postsyn_xyz = util.np_index_to_pixel_xyz(postsyn_xyz)
        # presyn_xyz = util.np_index_to_pixel_xyz(pred_graph[postsyn_id])
        print("%s <- %s" % (postsyn_xyz, pixel_coord))
    print("Extraction took {} seconds".format(time.time()- start_time))
    return pred_graph


def add_segmentation_labels(graph, zarr_path, dataset, materialize):
    print("Adding segmentation labels from {}".format(dataset))
    print("{} nodes to label".format(graph.number_of_nodes()))
    start_time = time.time()
    segment_array = daisy.open_ds(
        zarr_path,
        dataset)
    segment_array = segment_array[segment_array.roi]
    if materialize:
        print("Materializing")
        prediction_ds.materialize()
        print("Array materialized")
    nodes_outside_roi = []
    for i, (treenode_id, attr) in enumerate(graph.nodes(data=True)):
        try:
            attr['seg_label'] = int(segment_array[daisy.Coordinate(attr['zyx_coord'])])
        except AssertionError:
            nodes_outside_roi.append(treenode_id)
        if i == (graph.number_of_nodes() // 2):
            print("%s seconds remaining" % (time.time() - start_time))
    for node in nodes_outside_roi:
        graph.remove_node(node)
    print("Segmentation labels added in %s seconds" % (time.time() - start_time))
    return graph


def remove_intraneuron_synapses(pred_graph):
    counter = 0
    for presyn, postsyn in list(pred_graph.edges):
        presyn_neuron = pred_graph.nodes[presyn]['seg_label']
        postsyn_neuron = pred_graph.nodes[postsyn]['seg_label']

        if 'pixel_coord' in pred_graph.nodes[presyn]:
            presyn_xyz = pred_graph.nodes[presyn]['pixel_coord']
            postsyn_xyz = pred_graph.nodes[postsyn]['pixel_coord']
            print("%s (%d) <- %s (%d)" % (
                postsyn_xyz, postsyn_neuron, presyn_xyz, presyn_neuron
                ))
        if presyn_neuron == postsyn_neuron:
            pred_graph.remove_nodes_from((presyn, postsyn))
            counter += 1
    print("%s synapses between non distinct cells removed" % counter)
    return pred_graph


def remove_out_of_roi_synapses(graph, roi):
    counter = 0
    for presyn, postsyn in list(graph.edges):
        presyn_zyx = graph.nodes[presyn]['zyx_coord']
        postsyn_zyx = graph.nodes[postsyn]['zyx_coord']

        if (not roi.contains(presyn_zyx) or
                not roi.contains(postsyn_zyx)):
            graph.remove_nodes_from((presyn, postsyn))
            counter += 1

        # if 'pixel_coord' in graph.nodes[presyn]:
        #     presyn_xyz = graph.nodes[presyn]['pixel_coord']
        #     postsyn_xyz = graph.nodes[postsyn]['pixel_coord']
        #     print("%s (%d) <- %s (%d)" % (
        #         postsyn_xyz, postsyn_neuron, presyn_xyz, presyn_neuron
        #         ))
        # if presyn_neuron == postsyn_neuron:
    print("%s synapses outside of ROI removed" % counter)
    return graph


def remove_out_of_roi_nodes(graph, roi):
    nodes = []
    for n in list(graph.nodes):
        loc_zyx = graph.nodes[n]['zyx_coord']
        if not roi.contains(loc_zyx):
            nodes.append(n)
    graph.remove_nodes_from(nodes)
        # if 'pixel_coord' in graph.nodes[presyn]:
        #     presyn_xyz = graph.nodes[presyn]['pixel_coord']
        #     postsyn_xyz = graph.nodes[postsyn]['pixel_coord']
        #     print("%s (%d) <- %s (%d)" % (
        #         postsyn_xyz, postsyn_neuron, presyn_xyz, presyn_neuron
        #         ))
        # if presyn_neuron == postsyn_neuron:
    print("%s synapses outside of ROI removed" % len(nodes))
    return graph


def remove_nodes_of_type(graph, node_type):
    counter = 0
    nodes = []
    for presyn, postsyn in list(graph.edges):
        n = presyn if node_type == 'presyn' else postsyn
        nodes.append(n)
    graph.remove_nodes_from(nodes)

        # graph.remove_nodes_from((presyn, postsyn))
            # counter += 1

        # if 'pixel_coord' in graph.nodes[presyn]:
        #     presyn_xyz = graph.nodes[presyn]['pixel_coord']
        #     postsyn_xyz = graph.nodes[postsyn]['pixel_coord']
        #     print("%s (%d) <- %s (%d)" % (
        #         postsyn_xyz, postsyn_neuron, presyn_xyz, presyn_neuron
        #         ))
        # if presyn_neuron == postsyn_neuron:
    # print("%s synapses outside of ROI removed" % counter)
    return graph



