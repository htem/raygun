import json
import logging
import daisy
import sys
import math
import collections
import numpy as np
import gt_tools

from funlib.segment.arrays import replace_values

# sys.path.insert(0, "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/segway/tasks")
# import task_helper
# from fix_merge import get_graph, fix_merge

logging.basicConfig(level=logging.INFO)


def to_daisy_coord(xyz, voxel_size):
    return [xyz[2]*voxel_size[0], xyz[1]*voxel_size[1], xyz[0]*voxel_size[2]]


def to_pixel_coord(zyx):
    return [zyx[2]/voxel_size[2], zyx[1]/voxel_size[1], zyx[0]/voxel_size[0]]


def to_zyx(xyz):
    return [xyz[2], xyz[1], xyz[0]]


def segment_from_skeleton(skeletons, segment_array, nodes):
    # segments = collections.defaultdict(collections.defaultdict(list))
    segments = collections.defaultdict(lambda: collections.defaultdict(list))
    # print(nodes)

    for skeleton_id in skeletons:
        # print(skeletons[skeleton_id])
        if len(skeletons[skeleton_id]) < 3:
            # ignore skeletons with less than 2 nodes (namely the pre/post synaptic partner labels)
            continue

        for n in skeletons[skeleton_id]:
            zyx = daisy.Coordinate(tuple(nodes[n]["zyx"]))
            if not segment_array.roi.contains(zyx):
                continue
            segid = segment_array[zyx]
            assert(segid is not None)
            segments[segid][skeleton_id].append(n)

    # print(segments)
    # print(segments.keys())
    return segments


def interpolate_locations_in_z(zyx0, zyx1, voxel_size):
    z_res = voxel_size[0]

    # assert zyx0[0] % z_res == 0, "%d not divisible by voxel size %d" % (zyx0[0], z_res)
    # assert zyx1[0] % z_res == 0, "%d not divisible by voxel size %d" % (zyx1[0], z_res)
    if (zyx0[0] % z_res) or (zyx1[0] % z_res):
        # both locations need to be perfectly divisible to interpolate
        # for now
        return []
    steps = int(math.fabs(zyx1[0] - zyx0[0]) / z_res)
    if steps <= 1:
        return []

    delta = []
    for i in range(3):
        delta.append((float(zyx1[i]) - zyx0[i]) / steps)
    # print(delta)
    assert(int(delta[0]) == z_res or int(delta[0]) == -z_res)
    # for z in range(zyx0[2], zyx0[2], z_res):
    res = []
    for i in range(steps-1):
        res.append([int(zyx0[k] + (i+1)*delta[k]) for k in range(3)])

    return res


def parse_skeleton_json(json, voxel_size):
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

            if node_json["parent_id"] is not None:
                # need to check and make intermediate nodes
                # before appending current node
                prev_node_id = str(node_json["parent_id"])
                # print(skel_json["treenodes"])
                prev_node = skel_json["treenodes"][prev_node_id]
                intermediates = interpolate_locations_in_z(
                        to_zyx(prev_node["location"]),
                        node["zyx"],
                        voxel_size)
                for loc in intermediates:
                    int_node_id = len(nodes)
                    int_node = {"zyx": loc}
                    nodes[int_node_id] = int_node
                    skeleton.append(int_node_id)
            skeleton.append(node_id)

        # print(skeleton)
        # exit(0)
        skeletons[skel_id] = skeleton
    # print(skeletons)
    return skeletons, nodes


def get_all_segments_from_skeleton(segments):
    traced_segments = set()
    for s in segments:
        traced_segments.add(s)
    return list(traced_segments)


if __name__ == "__main__":
    '''Test case for fixing splitter'''

    config_f = sys.argv[1]
    config = gt_tools.load_config(config_f)

    raw_ds = daisy.open_ds(config["raw_file"], config["raw_ds"])

    voxel_size = raw_ds.voxel_size

    unlabeled_segments_zyx = []
    unlabeled_segments_xyz = config.get("unlabeled_segments_xyz", [])
    for xyz in unlabeled_segments_xyz:
        unlabeled_segments_zyx.append(daisy.Coordinate(to_daisy_coord(xyz, voxel_size)))

    foreground_segments_zyx = []
    foreground_segments_xyz = config.get("foreground_segments_xyz", [])
    for xyz in foreground_segments_xyz:
        foreground_segments_zyx.append(daisy.Coordinate(to_daisy_coord(xyz, voxel_size)))

    print("Making segment cache...")
    segment_file = config["segment_file"]
    # segment_dataset = config["segment_ds"]
    segment_dataset = config["segmentation_skeleton_ds"]
    segment_ds = daisy.open_ds(segment_file, segment_dataset, mode='r+')

    # skeleton_json = "skeleton.json"
    skeleton_json = config["skeleton_file"]
    with open(skeleton_json) as f:
        skeletons, nodes = parse_skeleton_json(json.load(f), segment_ds.voxel_size)

    segment_array = segment_ds[segment_ds.roi]
    segment_ndarray = segment_array.to_ndarray()
    segment_array = daisy.Array(
        segment_ndarray, segment_ds.roi, segment_ds.voxel_size)
    #segment_threshold = float(segment_dataset.split('_')[-1])

    segment_by_skeletons = segment_from_skeleton(
        skeletons, segment_array, nodes)

    # remove unlabeled segments
    for zyx in unlabeled_segments_zyx:
        segid = segment_array[zyx]
        if segid in segment_by_skeletons:
            del segment_by_skeletons[segid]
            # segment_by_skeletons.pop(segid)
            print("REMOVE %d" % segid)

    skeletonized_segments = get_all_segments_from_skeleton(
        segment_by_skeletons)

    unlabeled_ds = daisy.prepare_ds(
        segment_file,
        "volumes/labels/unlabeled_mask_skeleton",
        segment_ds.roi,
        segment_ds.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 5},
        delete=True
        )

    # unlabeled mask should be 0 in the context region
    print("Reset unlabeled mask...")
    unlabeled_ndarray = np.zeros(unlabeled_ds.shape, dtype=unlabeled_ds.dtype)

    skeletonized_segments = list(skeletonized_segments)
    new_mask_values = [1 for f in skeletonized_segments]
    replace_values(
        segment_ndarray,
        skeletonized_segments,
        new_mask_values,
        unlabeled_ndarray)

    # unlabeled_array[segment_ds.roi] = unlabeled_ndarray

    print("Write unlabeled mask...")
    unlabeled_ds[unlabeled_ds.roi] = unlabeled_ndarray

    if len(foreground_segments_zyx):

        print(foreground_segments_zyx)
        print(segment_array.roi)

        segment_by_foreground = [segment_array[zyx] for zyx in foreground_segments_zyx]

        unlabeled_ds = daisy.prepare_ds(
            segment_file,
            "volumes/labels/unlabeled_mask_skeleton_foreground",
            segment_ds.roi,
            segment_ds.voxel_size,
            np.uint8,
            compressor={'id': 'zlib', 'level': 5},
            delete=True
            )

        # unlabeled mask should be 0 in the context region
        print("Reset unlabeled mask...")
        unlabeled_ndarray = np.zeros(unlabeled_ds.shape, dtype=unlabeled_ds.dtype)

        # segment_by_skeletons_foreground = copy.deepcopy(segment_by_skeletons)
        # if len(foreground_segments_zyx):
        #     # remove all segment IDs not in this list
        #     for segid in segment_by_skeletons:
        #         if segid not in 

        segment_by_foreground = list(segment_by_foreground)
        new_mask_values = [1 for f in segment_by_foreground]
        replace_values(
            segment_ndarray,
            segment_by_foreground,
            new_mask_values,
            unlabeled_ndarray)

        print("Write foreground mask...")
        unlabeled_ds[unlabeled_ds.roi] = unlabeled_ndarray
