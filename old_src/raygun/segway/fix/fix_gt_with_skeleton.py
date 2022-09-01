import json
import logging
import daisy
import sys

import networkx
import numpy as np
import random

import task_helper

from funlib.evaluate import split_graph
from funlib.segment.arrays import replace_values_using_mask

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
# logging.getLogger('task_grow_segmentation').setLevel(logging.DEBUG)


def get_graph(input, threshold=0.9):
    graph = networkx.Graph()
    graph.add_nodes_from(input.nodes(data=True))

    for u, v, data in input.edges(data=True):
        assert(u in graph)
        assert(v in graph)
        if u not in graph or v not in graph:
            continue
        if (data['merge_score'] is not None and
                data['merge_score'] <= threshold):
            graph.add_edge(u, v, capacity=1.0-data['merge_score'])

    return graph


def fix_merge(
        components_zyx,
        fragments_file,
        fragments_dataset,
        db_host,
        db_name,
        edges_collection,
        segment_threshold,
        global_config,
        roi_offset=None,
        roi_shape=None,

        segment_file=None,
        segment_dataset=None,
        segment_array=None,
        **kwargs):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)
    print("Making fragment cache...")
    fragments_cached = fragments[fragments.roi].to_ndarray()
    fragments_cached = daisy.Array(
        fragments_cached, fragments.roi, fragments.voxel_size)
    # fragments_cached = fragments

    # open segment_ds
    if segment_array is None:
        assert(segment_file is not None and segment_dataset is not None)
        segment_ds = daisy.open_ds(segment_file, segment_dataset, mode='r+')
        print("Making segment cache...")
        segment_ds_cached = segment_ds[segment_ds.roi].to_ndarray()
        segment_ds_cached = daisy.Array(
            segment_ds_cached, segment_ds.roi, segment_ds.voxel_size)
    else:
        segment_ds_cached = segment_array
    # segment_ds_cached = segment_ds

    # open RAG DB
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode='r',
        edges_collection=edges_collection,
        position_attribute=['center_z', 'center_y', 'center_x'])

    components_zyx = [
        [daisy.Coordinate(tuple(zyx)) for zyx in zyxs]
        for zyxs in components_zyx
    ]

    components = [[fragments_cached[zyx] for zyx in comp] for comp in components_zyx]
    print("Components to be splitted %s" % components)

    # make sure that components are of different fragments
    fragments_ids = set()
    for comp in components:
        for n in comp:
            assert(n not in fragments_ids)
            fragments_ids.add(n)

    if roi_offset is not None and roi_shape is not None:
        total_roi = daisy.Roi(tuple(roi_offset), tuple(roi_shape))
    else:
        total_roi = fragments_cached.roi

    rag = rag_provider[total_roi]
    graph = get_graph(rag, segment_threshold)

    print("Input components: %s" % components)

    num_splits = split_graph(
        graph,
        components,
        position_attributes=['center_z', 'center_y', 'center_x'],
        weight_attribute="capacity",
        split_attribute="cut_id"
        )
    print("Num cuts made: %d" % num_splits)

    # relabeling split regions
    # assuming that all components given have the same segment_id
    merged_segment_id = segment_ds_cached[components_zyx[0][0]]
    parent_cut_id = graph.nodes[components[0][0]]["cut_id"]

    rand = int(random.random()*65536)

    # create new segment IDs
    new_segment_ids = {}
    for i in range(num_splits+1):
        if i == parent_cut_id:
            new_segment_ids[i] = merged_segment_id
        else:
            new_segment_ids[i] = rand+i

    # create remap list
    mask_values = []
    new_values = []

    rewrite_segment = True
    # rewrite_segment = False

    if rewrite_segment:
        print("Computing new segmentation...")
        for node, node_data in graph.nodes(data=True):
            node_zyx = daisy.Coordinate(tuple((node_data['center_z'],
                                        node_data['center_y'],
                                        node_data['center_x'])))
            if segment_ds_cached[node_zyx] == merged_segment_id:
                if node_data["cut_id"] != parent_cut_id:
                    # remap_list.append((node, new_segment_ids[node_data["cut_id"]]))
                    mask_values.append(node)
                    new_values.append(new_segment_ids[node_data["cut_id"]])

        mask_values = np.array(mask_values, dtype=fragments.dtype)
        new_values = np.array(new_values, dtype=segment_ds_cached.dtype)

        print("Replacing values...")
        new_segmentation = segment_ds_cached.to_ndarray()
        replace_values_using_mask(
            fragments_cached.to_ndarray(),
            mask_values,
            new_values,
            new_segmentation,
            inplace=True)

        if segment_ds is not None:
            # rewrite segment_file
            print("Writing new segmentation file...")
            segment_ds[segment_ds.roi] = new_segmentation


def to_daisy_coord(xyz):
    return [xyz[2]*40, xyz[1]*4, xyz[0]*4]


def to_pixel_coord(zyx):
    return [zyx[2]/4, zyx[1]/4, zyx[0]/40]


if __name__ == "__main__":
    '''Test case for fixing splitter'''

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])
    skeleton_json = "skeleton.json"

    with open(skeleton_json) as f:
        skeletons = parse_skeleton_json(json.load(f))

    # segment_file = global_config["Input"]["output_file"]
    # segment_dataset = "volumes/segmentation_0.900"

    # segment_ds = daisy.open_ds(segment_file, segment_dataset, mode='r+')
    # segment_array = segment_ds[segment_ds.roi].to_ndarray()
    # segment_threshold = float(segment_dataset.split('_')[-1])

    fix_merge(
        components_zyx=components_zyx,
        segment_array=segment_array,
        global_config=global_config,
        **user_configs, **global_config["FixMergeTask"])
