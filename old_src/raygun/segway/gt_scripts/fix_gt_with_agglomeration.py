import json
import logging
import daisy
import sys
import math
import collections
import json
import os

import networkx

# import numpy as np
import gt_tools

from funlib.segment.arrays import replace_values

from fix_merge_with_agglomeration import fix_merge2


def get_graph(
        input, threshold, rag_weight_attribute="capacity", segment_ds=None,
        filter_in_segments=[],
        components=None,
        ):
    graph = networkx.Graph()
    for n, n_data in input.nodes(data=True):

        if 'center_z' not in n_data:
            continue

        if 'segment_id' not in n_data and segment_ds is not None:

            zyx = daisy.Coordinate(tuple([n_data[c] for c in position_attributes]))

            if not segment_ds.roi.contains(zyx):
                continue

            segment_id = segment_ds[zyx]
            n_data["segment_id"] = segment_id

        if len(filter_in_segments) and \
                n_data['segment_id'] not in filter_in_segments:
            continue

        graph.add_node(n, **n_data)

    component_by_id = {}
    if components is not None:
        for component_id, nodes in enumerate(components):
            for n in nodes:
                component_by_id[n] = component_id

    for u, v, data in input.edges(data=True):

        if u not in graph or v not in graph:
            continue

        if u in component_by_id and v in component_by_id:
            if component_by_id[u] == component_by_id[v]:
                merge_score = .01
                capacity = 100
            else:
                merge_score = .99
                capacity = .01
            graph.add_edge(
                u, v,
                capacity=capacity,
                merge_score=merge_score,
                )

        elif (data['merge_score'] is not None and
                data['merge_score'] <= threshold):
            graph.add_edge(
                u, v,
                capacity=1.0-data['merge_score'],
                merge_score=data['merge_score'],
                )

    return graph


# try:
#     import graph_tool
# except ImportError:
#     print("Error: graph_tool is not found.")
#     exit(0)

logging.basicConfig(level=logging.INFO)

# global voxel_size
# voxel_size = [40, 4, 4]

def to_daisy_coord(xyz):
    global voxel_size
    return [xyz[2]*voxel_size[0],
            xyz[1]*voxel_size[1],
            xyz[0]*voxel_size[2],
            ]

def to_pixel_coord(zyx):
    global voxel_size
    return [
            zyx[2]/voxel_size[2],
            zyx[1]/voxel_size[1],
            zyx[0]/voxel_size[0],
            ]


def to_zyx(xyz):
    return [xyz[2], xyz[1], xyz[0]]


def segment_from_skeleton(
        skeletons,
        segment_array,
        nodes,
        fragments_array,
        merge_correction_lut):

    # print("Getting segments from skeletons...")

    segments = collections.defaultdict(lambda: collections.defaultdict(list))

    # print(merge_correction_lut)

    for skeleton_id in skeletons:
        for node in skeletons[skeleton_id]:
            zyx = daisy.Coordinate(tuple(nodes[node]["zyx"]))
            if not segment_array.roi.contains(zyx):
                continue

            fid = fragments_array[zyx]
            if fid in merge_correction_lut:
                segid = merge_correction_lut[fid]
                # print("%d: %d" % (fid, segid))
            else:
                segid = segment_array[zyx]

            assert(segid is not None)
            segments[segid][skeleton_id].append(node)

    return segments


def get_one_merged_components(segments, done):
    for s in segments:
        if s in done:
            continue
        components = segments[s]
        if len(components) > 1:
            return (s, [components[c] for c in components])
    return (None, None)


def get_one_splitted_component(
        skeletons, segment_array, nodes, done,
        fragments_array, ignored_fragments,
        split_correction_lut,
        ):

    for skid in skeletons:
        segments = []
        zyxs = []
        if skid in done:
            continue
        s = skeletons[skid]
        for n in s:
            zyx = daisy.Coordinate(tuple(nodes[n]["zyx"]))
            if not segment_array.roi.contains(zyx):
                continue

            fid = fragments_array[zyx]
            if fid in ignored_fragments:
                continue

            seg_id = segment_array[zyx]

            if seg_id in split_correction_lut:
                seg_id = split_correction_lut[seg_id]
            if seg_id != 0:
                # TODO: not entirely sure why this could be
                # when bound is checked above
                segments.append(seg_id)
                zyxs.append(zyx)
        if len(segments) > 1:
            return (segments, skid, zyxs)
    return (None, None, None)


def interpolate_locations_in_z(zyx0, zyx1):
    # assuming z pixel size is 40
    assert(zyx0[0] % 40 == 0)
    assert(zyx1[0] % 40 == 0)
    steps = int(math.fabs(zyx1[0] - zyx0[0]) / 40)
    if steps <= 1:
        return []

    delta = []
    for i in range(3):
        delta.append((float(zyx1[i]) - zyx0[i]) / steps)
    assert(int(delta[0]) == 40 or int(delta[0]) == -40)
    res = []
    for i in range(steps-1):
        res.append([int(zyx0[k] + (i+1)*delta[k]) for k in range(3)])

    return res


def parse_skeleton_json(json, interpolation=True):
    skeletons = {}
    nodes = {}
    json = json["skeletons"]
    for skel_id in json:
        skel_json = json[skel_id]
        # skeletons[skel_id] = []
        skeleton = []
        for node_id_json in skel_json["treenodes"]:
            node_json = skel_json["treenodes"][node_id_json]
            node = {"zyx": to_zyx(node_json["location"])}
            node_id = len(nodes)
            nodes[node_id] = node

            if interpolation:
                if node_json["parent_id"] is not None:
                    # need to check and make intermediate nodes
                    # before appending current node
                    prev_node_id = str(node_json["parent_id"])
                    prev_node = skel_json["treenodes"][prev_node_id]
                    intermediates = interpolate_locations_in_z(
                        to_zyx(prev_node["location"]), node["zyx"])
                    for loc in intermediates:
                        int_node_id = len(nodes)
                        int_node = {"zyx": loc}
                        nodes[int_node_id] = int_node
                        skeleton.append(int_node_id)

            skeleton.append(node_id)

        if len(skeleton) == 1:
            # skip single node skeletons (likely to be synapses annotation)
            continue
        
        skeletons[skel_id] = skeleton
    return skeletons, nodes


if __name__ == "__main__":

    global voxel_size

    config = gt_tools.load_config(sys.argv[1])
    file = config["file"]

    correct_merges = True
    if len(sys.argv) > 2 and sys.argv[2] == "--no_correct_merge":
        correct_merges = False
    correct_splits = True
    if len(sys.argv) > 2 and sys.argv[2] == "--no_correct_split":
        correct_splits = False
    make_segment_cache = True
    make_fragment_cache = True

    segmentation_skeleton_ds = config["segmentation_skeleton_ds"]
    segment_dataset = config["segment_ds"]

    reagglomeration_threshold = config.get("fix_gt_agglomeration_threshold", None)
    # try to guess from segmentation_skeleton_ds
    if reagglomeration_threshold is None:
        try:
            reagglomeration_threshold = float(segmentation_skeleton_ds.split('_')[-1])
        except:
            pass
    if reagglomeration_threshold is None:
        try:
            reagglomeration_threshold = float(segment_dataset.split('_')[-1])
        except:
            pass
    if reagglomeration_threshold is None:
        reagglomeration_threshold = 0.9

    print("reagglomeration_threshold:", reagglomeration_threshold)

    segment_file = config.get("segment_file", config["file"])
    segment_ds = daisy.open_ds(segment_file, segment_dataset)

    voxel_size = segment_ds.voxel_size
    interpolation = True
    if voxel_size[0] == voxel_size[1] and voxel_size[0] == voxel_size[2]:
        interpolation = False

    skeleton_json = config["skeleton_file"]
    with open(skeleton_json) as f:
        skeletons, nodes = parse_skeleton_json(json.load(f), interpolation=interpolation)

    # print(skeletons); exit(0)
    zyx_components = []
    for sid in skeletons:
        skel = skeletons[sid]
        component = []
        for nid in skel:
            component.append(nodes[nid]['zyx'])
        zyx_components.append(component)
    # print(zyx_components); exit(0)

    segment_array = segment_ds[segment_ds.roi]
    if make_segment_cache:
        print("Making segment cache...")
        segment_ndarray = segment_array.to_ndarray()
        segment_array = daisy.Array(
            segment_ndarray, segment_ds.roi, segment_ds.voxel_size)

    segment_threshold = float(config["segment_ds"].split('_')[-1])

    fragments_file = config.get("fragments_file", config["file"])
    fragments_dataset = config.get("fragments_ds", 'volumes/fragments')
    fragments_ds = daisy.open_ds(fragments_file, fragments_dataset)
    fragments_array = fragments_ds[fragments_ds.roi]

    affs_file = config.get("affs_file", config["file"])
    affs_dataset = config.get("affs_ds", 'volumes/affs')
    affs_ds = daisy.open_ds(affs_file, affs_dataset)
    affs_array = affs_ds[affs_ds.roi]

    problem_fragments_filename = os.path.join(segment_file, "problem_fragments.json")

    print("Creating corrected segment at %s" % segmentation_skeleton_ds)

    corrected_segment_ds = daisy.prepare_ds(
        segment_file,
        segmentation_skeleton_ds,
        segment_ds.roi,
        segment_ds.voxel_size,
        segment_ds.dtype,
        compressor={'id': 'zlib', 'level': 5}
        )

    merge_correction_lut = {}

    if correct_merges:

        if make_fragment_cache:
            print("Making fragment cache...")
            fragments_array.materialize()

        next_segid = None

        ignored_fragments = []
        if "ignored_fragments" in config:
            print("Processing ignored_fragments...")
            ignored_fragments = config["ignored_fragments"]

        print("Correcting merges...")
        errored_fragments = []

        fix_merge2(
            zyx_components,
            affs_array,
            fragments_array,
            segment_array,
            # rag,
            # rag_weight_attribute,
            ignored_fragments=ignored_fragments,
            next_segid=next_segid,
            errored_fragments_out=errored_fragments,
            # fragments_lut=merge_correction_lut,
            reagglomeration_threshold=reagglomeration_threshold
            )


        print("Problem fragments with more than one skeleton nodes:")
        if len(errored_fragments):
            for f in errored_fragments:
                print(f)

        for f, coord in errored_fragments:
            ignored_fragments.append(f)
        ignored_fragments = set(ignored_fragments)

        with open(problem_fragments_filename, 'w') as f:
            ignored_fragments_json = [str(k) for k in ignored_fragments]
            json.dump(ignored_fragments_json, f)

    if correct_splits:
        processed_segments = set()

        if make_fragment_cache:
            print("Making fragment cache...")
            fragments_array.materialize()

        split_correction_lut = {}

        while True:
            splitted_segments, skeleton_id, coords = get_one_splitted_component(
                    skeletons,
                    segment_array,
                    nodes,
                    processed_segments,
                    fragments_array,
                    ignored_fragments,
                    split_correction_lut,
                    # merge_correction_lut
                    )
            processed_segments.add(skeleton_id)

            if splitted_segments is None:
                break  # done

            # print("Splitted segments:")
            # for s, zyx in zip(splitted_segments, coords):
            #     # print("Splitted segments: %s" % splitted_segments)
            #     print("%s (%s)" % (s, to_pixel_coord(zyx)), end=', ')
            # print()

            root_label = splitted_segments[0]
            while root_label in split_correction_lut:
                root_label = split_correction_lut[root_label]

            for s in splitted_segments:
                if s != root_label:
                    split_correction_lut[s] = root_label

        print("Writing split corrections...")
        mask_values = []
        new_values = []
        for k in split_correction_lut:
            mask_values.append(k)
            new_values.append(split_correction_lut[k])
        replace_values(
            segment_array.data,
            mask_values,
            new_values,
            segment_array.data,
            )

    print("Write segmentation to disk...")
    corrected_segment_ds[corrected_segment_ds.roi] = segment_array.to_ndarray()
