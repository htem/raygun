import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc
from extract_segId_from_prediction import add_predicted_seg_labels_from_vol, \
                                          replace_fragment_ids_with_LUT_values
from build_graph_from_catmaid import construct_skeleton_graph
from evaluation_matrix import find_merge_errors, find_split_errors, rand_voi_split_merge
import re
import daisy
from daisy import Coordinate
from multiprocessing import Pool
from functools import partial
import numpy as np
import networkx as nx
import os
import csv
import copy
from itertools import product
import random


def compare_segmentation_to_ground_truth_skeleton(
        agglomeration_thresholds,
        segmentation_paths,
        model_name_mapping,
        num_processes,
        configs):
    split_and_merge, split_and_merge_rand, split_and_merge_voi = [], [], []
    colours = color_generator(len(segmentation_paths))

    parameters = [
        (agglomeration_thresholds, seg_path, num_processes, configs['skeleton'])
            for seg_path in segmentation_paths
        ]

    p = Pool(num_processes)
    graph_lists = p.starmap(generate_graphs_with_seg_labels,parameters)

    for i,seg_path in enumerate(segmentation_paths):
        numb_split, numb_merge = [], []
        (rand_split_list,
         rand_merge_list,
         voi_split_list,
         voi_merge_list) = [], [], [], []

        graph_list= graph_lists[i]
        for graph in graph_list:
            if graph is None:
                numb_split.append(np.nan)
                numb_merge.append(np.nan)
                rand_split_list.append(np.nan)
                rand_merge_list.append(np.nan)
                voi_split_list.append(np.nan)
                voi_merge_list.append(np.nan)
            else:
                error_configs, output_configs = configs['error_count'], configs['output']
                split_errors, _= find_split_errors(graph,
                                                   error_configs['ignore_glia'],
                                                   error_configs['max_break_size'])
                merge_errors, _ = find_merge_errors(graph,
                                                    error_configs['z_weight_multiplier'],
                                                    error_configs['ignore_glia'],
                                                    error_configs['assume_minimal_merges'])
                if output_configs['write_TXT'] or output_configs['write_CSV']:
                    seg_vol = agglomeration_thresholds[graph_list.index(graph)]
                    merge_error_rows, split_error_rows = format_errors(merge_errors, split_errors, graph, output_configs['voxel_size'])
                    output_path, file_name = generate_output_path_and_file_name(output_configs, seg_vol, seg_path)
                    write_error_files(output_path, file_name,
                                      merge_error_rows, split_error_rows,
                                      output_configs['write_TXT'],
                                      output_configs['write_CSV'])
                (rand_split, rand_merge,
                voi_split, voi_merge) = rand_voi_split_merge(graph)
                numb_split.append(len(split_errors))
                numb_merge.append(len(merge_errors))
                rand_split_list.append(rand_split)
                rand_merge_list.append(rand_merge)
                voi_split_list.append(voi_split)
                voi_merge_list.append(voi_merge)
        model = get_model_name(seg_path, model_name_mapping)
        split_and_merge.extend((model, numb_merge, numb_split))
        split_and_merge_rand.extend((model, rand_merge_list, rand_split_list))
        split_and_merge_voi.extend((model, voi_merge_list, voi_split_list))

    plot_errors = partial(generate_error_plot,
                          agglomeration_thresholds,
                          configs['output']['config_JSON'],
                          configs['name'],
                          configs['output']['output_path'],
                          configs['output']['markers'],
                          colours,
                          configs['output']['font_size'],
                          configs['output']['line_width'])
    plot_errors('number', *split_and_merge)
    plot_errors('rand', *split_and_merge_rand)
    plot_errors('voi', *split_and_merge_voi)

    split_and_merge = {configs['name'] : split_and_merge}

    return(split_and_merge)

def color_generator(length):
    random.seed(3)
    more_colour_vals= []
    while(len(more_colour_vals)!=length):
        r = random.random()
        if r not in more_colour_vals: more_colour_vals.append(r)
    more_colours=cm.jet(more_colour_vals)

    more_colours = ["blue", "yellow","black","coral","purple","navy","lime","red","fuchsia","gold", "maroon", "darkgreen", "darkslategrey", "fuchsia"]

    return(more_colours)



def generate_graphs_with_seg_labels(agglomeration_thresholds, segmentation_path,num_processes, skeleton_configs):
    graph_list = []
    unlabelled_skeleton = construct_skeleton_graph(skeleton_configs['skeleton_path'],skeleton_configs['with_interpolation'],skeleton_configs['step'],skeleton_configs['leaf_node_removal_depth'])

    if os.path.exists(os.path.join(segmentation_path, 'luts/fragment_segment')):
        fragment_graph = add_predicted_seg_labels_from_vol(unlabelled_skeleton.copy(),segmentation_path,'volumes/fragments',skeleton_configs['load_segment_array_to_memory'])

        parameters_list = [(fragment_graph.copy(), segmentation_path, 'volumes/'+threshold)
                            for threshold in agglomeration_thresholds]

        graph_list = [replace_fragment_ids_with_LUT_values(pl[0],pl[1],pl[2]) for pl in parameters_list]
    else:
        parameters_list = [(unlabelled_skeleton.copy(), segmentation_path, 'volumes/'+threshold,
                            skeleton_configs['load_segment_array_to_memory'])
                            for threshold in agglomeration_thresholds]
        graph_list= [add_predicted_seg_labels_from_vol(pl[0],pl[1],pl[2],pl[3]) for pl in parameters_list]

    return graph_list



def generate_error_plot(
        agglomeration_thresholds,
        config_file_name,
        volume_name,
        output_path,
        markers,
        colors,
        font_size,
        line_width,
        error_metric,
        *split_and_merge):

    fig, ax = plt.subplots(figsize=(8, 6))
    for j in range(int(len(split_and_merge)/3)):

        ax.plot(split_and_merge[j*3+1], split_and_merge[j*3+2],
                label=split_and_merge[j*3], color = colors[j],
                zorder=1, alpha=0.8, linewidth=line_width)
        for a, b, m, l in zip(split_and_merge[j*3+1], split_and_merge[j*3+2],
                              markers, agglomeration_thresholds):
            if j == 0:
                ax.scatter(a, b, marker=m,  color =colors[j],
                           label=l.replace('segmentation_', ''),
                           zorder=2, alpha=.9, s=45)
            else:
                ax.scatter(a, b, marker=m,  color = colors[j],zorder=2, alpha=0.5,
                           s=30)
    ax.legend(prop={'size': font_size})
    if error_metric == 'number':
        #ax.set_ylim(bottom=-0.8)
        #ax.set_xlim(left=-0.8)
        plt.xlabel('Merge Error Count')
        plt.ylabel('Split Error Count')
    elif error_metric == 'rand':
        #ax.set_ylim(bottom=0)
        #ax.set_xlim(left=-0.01)
        plt.xlabel('Merge Rand')
        plt.ylabel('Split Rand')
    elif error_metric == 'voi':
        #ax.set_ylim(bottom=0)
        #ax.set_xlim(left=-0.01)
        plt.xlabel('Merge VOI')
        plt.ylabel('Split VOI')

    output_file_name = output_path+'/'+config_file_name+'_'+volume_name+'_'+error_metric

    plt.savefig(output_file_name, dpi=300)


def get_model_name(volume_path, name_dictionary={}):
    if volume_path in name_dictionary:
        return name_dictionary[volume_path]
    if re.search(r'setup[0-9]{2}', volume_path):
        model = re.search(r'setup[0-9]{2}',
                          volume_path).group(0) + \
                          '_'+re.search(r'[0-9]+00',
                                        volume_path).group(0)
        return model


def write_error_files(output_path, file_name,
                      merge_error_rows, split_error_rows,
                      write_txt, write_csv):
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    print("output path:", output_path)
    if write_txt:
        with open(output_path + file_name + '.txt', 'w') as f:
            print("MERGE ERRORS (" + str(len(merge_error_rows)) + ")", file=f)
            for error in merge_error_rows:
                print("Segment %s" % error[1], file = f)
                print("\t%s and %s merged" % (error[2], error[3]), file = f)
                print("\tCATMAID nodes %s and %s" % (error[4], error[5]), file = f)
                print(file = f)
            print("SPLIT_ERRORS (" + str(len(split_error_rows)) + ")", file = f)
            for error in split_error_rows:
                print("Skeleton %s" % error[1], file = f)
                print("\t%s and %s split" % (error[2], error[3]), file = f)
                print("\tCATMAID nodes %s and %s" % (error[4], error[5]), file = f)

                print(file = f)
    if write_csv:
        with open(output_path + file_name + '.csv', 'w') as f:
            csvwriter = csv.writer(f)
            fields = ["error type", "ID (segment if merge, skeleton if split)",
                      "coordinate 1", "coordinate 2", "node 1", "node 2"]
            csvwriter.writerow(fields)
            csvwriter.writerows(merge_error_rows)
            csvwriter.writerows(split_error_rows)


def format_errors(merge_errors, split_errors, graph, voxel_size):
    merge_error_rows, split_error_rows = [], []
    for error in merge_errors:
        node1, node2 = error[0], error[1]
        segment_id = str(graph.nodes[node1]['seg_label'])
        coord1 = str(to_pixel_coord_xyz(graph.nodes[node1]['zyx_coord'], voxel_size))
        coord2 = str(to_pixel_coord_xyz(graph.nodes[node2]['zyx_coord'], voxel_size))
        merge_error_rows.append(['M', segment_id, coord1, coord2, node1, node2])
    for error in split_errors:
        node1, node2 = error[0], error[1]
        skeleton_id = str(graph.nodes[node1]['skeleton_id'])
        coord1 = str(to_pixel_coord_xyz(graph.nodes[node1]['zyx_coord'], voxel_size))
        coord2 = str(to_pixel_coord_xyz(graph.nodes[node2]['zyx_coord'], voxel_size))
        split_error_rows.append(['S', skeleton_id, coord1, coord2, node1, node2])
    return merge_error_rows, split_error_rows


def generate_output_path_and_file_name(output_configs, seg_vol, seg_path):
    output_path = os.path.join(output_configs['output_path'],
                               output_configs['config_JSON'] + '_error_coords/')
    seg_info = seg_path.split('/')
    file_name = 'error_coords_' + seg_info[-4] + '_' + \
            seg_info[-3] + '_' + seg_info[-2] + '_' + seg_vol
    return output_path, file_name


# following code is to find the coordinate of split or merge error
def to_pixel_coord_xyz(zyx, voxel_size):
    zyx = (Coordinate(zyx) / Coordinate(voxel_size))
    return Coordinate((zyx[2], zyx[1], zyx[0]))
