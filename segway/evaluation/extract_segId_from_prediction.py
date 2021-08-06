import daisy
import time
import json
import os.path
import networkx as nx
import numpy as np
        

def add_predicted_seg_labels_from_vol(graph, segmentation_path, segment_dataset, load_segment_array_to_memory = True):
    start_time = time.time()
    print('Adding segmentation predictions from %s' % segment_dataset)
    segment_array = daisy.open_ds(
        segmentation_path,
        segment_dataset)
    segment_array = segment_array[segment_array.roi]
    if load_segment_array_to_memory:
        segment_array.materialize()
        print("Segment array materialized")
    nodes_outside_roi = []  
    for i, (treenode_id, attr) in enumerate(graph.nodes(data=True)):
        try:    
            attr['zyx_coord'] = (attr['z'], attr['y'], attr['x'])
            attr['seg_label'] = segment_array[daisy.Coordinate(attr['zyx_coord'])]
        except AssertionError:
            nodes_outside_roi.append(treenode_id)
        if i % 1000 == 0:
            print("%s of %s nodes labelled" % (i, graph.number_of_nodes()))
    for node in nodes_outside_roi:
        graph.remove_node(node)
    if segment_dataset == 'volumes/fragments':
        return graph
    else:
        return assign_skeleton_indexes(graph)


def replace_fragment_ids_with_LUT_values(fragment_graph, segmentation_path, segment_dataset):
    print(segmentation_path)
    start_time = time.time()
    print('Adding segmentation predictions from %s look up tables' % segment_dataset)
    nodes = list(fragment_graph.nodes)
    local_lut, global_lut = load_lut(segmentation_path, segment_dataset)
    for i, treenode_id in enumerate(nodes):
        attr = fragment_graph.nodes[treenode_id]
        frag_id = attr['seg_label']
        if frag_id == 0:
            fragment_graph.remove_node(treenode_id)
        else:
            local_lut_index = np.nonzero(local_lut[0, :] == frag_id)[0].item()
            local_label = local_lut[1, local_lut_index].item()
            global_lut_index = np.nonzero(global_lut[0, :] == local_label)[0].item()
            global_label = global_lut[1, global_lut_index].item()
            attr['seg_label'] = global_label
    print('Task add_segId_from_prediction of %s took %s seconds' %
          (segment_dataset, round(time.time()-start_time, 3)))
    return assign_skeleton_indexes(fragment_graph)


def load_lut(segmentation_path, segment_dataset):
    agglomeration_percentile = str(int(segment_dataset.split('.')[-1]) // 10)
    lut_dir = segmentation_path+'/luts/fragment_segment/'
    local_lut_path = os.path.join(lut_dir, 'seg_frags2local_hist_quant_50_'+agglomeration_percentile)
    partial_local_luts = [np.load(os.path.join(local_lut_path, partial_lut))['fragment_segment_lut']
                          for partial_lut in os.listdir(local_lut_path) if partial_lut.endswith('.npz')]
    local_lut = np.concatenate(partial_local_luts, axis=1)

    global_lut_path = os.path.join(lut_dir, 'seg_local2global_hist_quant_50_'+agglomeration_percentile)
    if os.path.exists(global_lut_path+'_single.npz'):
        global_lut = np.load(global_lut_path+'_single.npz')['fragment_segment_lut']
    # NOTE: The LUT table doesn't seem to work when it's constructed
    # by concatenating partial tables in the folder
    else:
        partial_global_luts = [np.load(os.path.join(global_lut_path, partial_lut))['fragment_segment_lut']
                               for partial_lut in os.listdir(global_lut_path) if partial_lut.endswith('.npz')]
        global_lut = np.concatenate(partial_global_luts, axis=1)
    return local_lut, global_lut


# Assign unique ids to each cluster of connected nodes. This is to
# differentiate between sets of nodes that are discontinuous in the
# ROI but actually belong to the same skeleton ID, which is necessary
# because the network should not be penalized for incorrectly judging
# that these processes belong to different neurons.
def assign_skeleton_indexes(graph):
    skeleton_index_to_id = {}
    skel_clusters = nx.connected_components(graph)
    for i, cluster in enumerate(skel_clusters):
        for node in cluster:
            graph.nodes[node]['skeleton_index'] = i
        skeleton_index_to_id[i] = graph.nodes[cluster.pop()]['skeleton_id']
        
    return graph

