import daisy
import json
import logging
import sys
# import time
import os
import copy

from collections import defaultdict

import pymongo
import numpy as np

from filedb_graph import FileDBGraph
# import task_helper

logger = logging.getLogger(__name__)
# np.set_printoptions(threshold=sys.maxsize, formatter={'all':lambda x: str(x)})


# def load_frags2local_lut(block_id, merge_function, threshold, lut_dir):
#     lut = 'seg_frags2local_%s_%d/%d.npz' % (merge_function, int(threshold*100), block_id)
#     lut = os.path.join(
#             lut_dir,
#             lut)
#     if not os.path.exists(lut):
#         logging.info("Skipping %s of %s.." % (lut, block_id))
#         return None
#     lut = np.load(lut)['fragment_segment_lut']
#     return lut


# def write_interthreshold_connected_components(
#         block_id, merge_function, connected_components, source_threshold, target_threshold, 
#         lut_dir, debug=False):

#     if not debug:
#         fname = 'threshold_map_%s_%d_%d/%d.npz' % (
#             merge_function, int(source_threshold*100), int(target_threshold*100), block_id)
#     else:
#         fname = 'threshold_map_%s_%d_%d/debug/%d.npz' % (
#             merge_function, int(source_threshold*100), int(target_threshold*100), block_id)
#     fname = os.path.join(lut_dir, fname)
#     np.savez_compressed(fname, connected_components=connected_components)


def getHierarchicalMeshPath(object_id, hierarchical_size=10000):

    assert object_id != 0

    level_dirs = []
    num_level = 0
    while object_id > 0:
        level_dirs.append(int(object_id % hierarchical_size))
        object_id = int(object_id / hierarchical_size)
    num_level = len(level_dirs) - 1
    level_dirs = [str(lv) for lv in reversed(level_dirs)]
    return os.path.join(str(num_level), *level_dirs)


def exists_mesh(object_id):

    dir = "/n/pure/htem/Segmentation/cb2_v4/output.zarr/meshes/precomputed/mesh"
    path = os.path.join(dir, getHierarchicalMeshPath(object_id))
    exists = os.path.exists(path)
    # if not exists:
    #     print("not exists", path)
    # else:
    #     print("exists", path)
    return exists


def worker_function(
        # out_file,
        # lut_dir,
        # merge_function,
        # roi_offset,
        # roi_shape,
        # total_roi,
        local_graph,
        subseg_graph,
        source_threshold,
        thresholds,
        block,
        ):

    logger.info("Processing %s" % block)

    # lut_dir = os.path.join(
    #     out_file,
    #     lut_dir)

    # base_lut = load_frags2local_lut(block_id, merge_function, source_threshold, lut_dir)
    # base_lut = subseg_graph.load
    base_lut = local_graph.load_attribute(
        "seg_frags2local", block, source_threshold, attr_name="fragment_segment_lut")
    if base_lut is None:
        # file from previous step doesn't exist; we assume the previous step is correct
        return

    skipped_mesh_count = 0
    total_mesh_count = 0

    base_lut_dict = dict()
    # # optimization: we only need to have one fragment per subsegment
    processed_locals = set()
    for fragment, local in zip(base_lut[0], base_lut[1]):
        if local in processed_locals:
            continue
        if local == 0:
            continue
        total_mesh_count += 1
        if not exists_mesh(local):
            processed_locals.add(local)
            skipped_mesh_count += 1
            continue
        base_lut_dict[fragment] = local
        processed_locals.add(local)

    print("skipped_mesh_count: %d/%d" % (skipped_mesh_count, total_mesh_count))

    for target_threshold in thresholds:

        # lut = load_frags2local_lut(block_id, merge_function, target_threshold, lut_dir)
        lut = local_graph.load_attribute(
            "seg_frags2local", block, target_threshold, attr_name="fragment_segment_lut")
        assert lut is not None

        connected_components = defaultdict(set)

        for fragment, local in zip(lut[0], lut[1]):
            if fragment in base_lut_dict:
                base_seg = base_lut_dict[fragment]
                connected_components[local].add(base_seg)

        # print(connected_components); exit()

        # for k in connected_components:
        #     cc = copy.deepcopy(connected_components[k])
        #     cc.add(k)
        #     cc = list(cc)

        # connected_components_out_debug = []
        connected_components_out = []
        for k in connected_components:
            cc = connected_components[k]
            cc.add(k)
            if len(cc) == 1:
                continue
            connected_components_out.append(list(cc))
        # print(connected_components_out); exit()

        subseg_graph.write_attribute(connected_components_out, "threshold_map", block, target_threshold)
        # subseg_graph.write_attribute(connected_components_out_debug, "threshold_map_debug", block, target_threshold)

        # write_interthreshold_connected_components(
        #     block_id, merge_function, connected_components_out, source_threshold, target_threshold, lut_dir)

        # write_interthreshold_connected_components(
        #     block_id, merge_function, connected_components_out_debug, source_threshold, target_threshold, lut_dir, debug=True)


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    if run_config.get('block_id_add_one_fix', False):
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]

    local_graph = FileDBGraph(
        filepath=local_graph_dir,
        blocksize=local_block_size,
        roi_offset=file_graph_roi_offset,
        attr_postpend=merge_function,
        )

    subseg_graph = FileDBGraph(
        filepath=subseg_graph_dir,
        blocksize=super_block_size,
        roi_offset=file_graph_roi_offset,
        )

    # total_roi = daisy.Roi(total_roi_offset, total_roi_shape)

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        # roi_offset = block.write_roi.get_offset()
        # roi_shape = daisy.Coordinate(tuple(block_size))

        worker_function(
            local_graph=local_graph,
            subseg_graph=subseg_graph,
            # out_file=out_file,
            # lut_dir=lut_dir,
            # merge_function=merge_function,
            # total_roi=total_roi,
            # roi_offset=roi_offset,
            # roi_shape=roi_shape,
            source_threshold=source_threshold,
            thresholds=thresholds,
            block=block,
            )

        # recording block done in the database
        document = dict()
        document.update({
            'block_id': block.block_id,
            'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
            'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
            'start': 0,
            'duration': 0
        })
        completion_db.insert(document)

        client_scheduler.release_block(block, ret=0)
