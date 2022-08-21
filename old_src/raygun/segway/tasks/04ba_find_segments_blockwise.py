import daisy
import os
import json
import logging
import sys
import time
import numpy as np
import pymongo

from funlib.segment.graphs.impl import connected_components

# import task_helper
from task_04d_find_segment_blockwise import enumerate_blocks_in_chunks

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)


def get_chunkwise_lut(
        block,
        super_block_size,
        total_roi,
        lut_dir_local,
        lut_dir_subseg,
        merge_function,
        thresholds,
        super_chunk_size
        ):
    '''Compute sugsegment edges (to local) and nodes.
    Necessary for auto-grow in dahlia.'''

    print("block:", block)
    # print("super_block_size:", super_block_size)
    # print("super_chunk_size:", super_chunk_size)
    # print("total_roi:", total_roi)
    blocks = enumerate_blocks_in_chunks(
        block, super_block_size, super_chunk_size, total_roi)

    base_lut_dir_out = lut_dir_subseg
    for threshold in thresholds:
        lut_dir_subseg = base_lut_dir_out + '_%d' % int(threshold*100)

        local_nodes_list = []
        for b in blocks:
            # print("Loading", b)
            nodes_file = 'nodes_%s_%d/%d.npz' % (
                merge_function, int(threshold*100), b.block_id)
            nodes_file = os.path.join(lut_dir_local, nodes_file)
            try:
                local_nodes_list.append(np.load(nodes_file)['nodes'])
            except:
                # File not found maybe due to degenerate segmentation
                # (all sections missing raw at the edge of vol)
                # TODO: better error detection
                pass
        nodes = np.concatenate(local_nodes_list)

        local_edges_list = []
        for b in blocks:
            edges_file = 'edges_local2local_%s_%d/%d.npz' % (
                merge_function, int(threshold*100), b.block_id)
            edges_file = os.path.join(lut_dir_local, edges_file)
            try:
                local_edges_list.append(np.load(edges_file)['edges'])
            except:
                # File not found maybe due to degenerate segmentation
                # (all sections missing raw at the edge of vol)
                # TODO: better error detection
                pass
        edges = np.concatenate(local_edges_list)

        # print("Getting CCs...")
        # start = time.time()
        if len(edges):
            components = connected_components(
                                        nodes, edges,
                                        scores=None, threshold=1.0,
                                        no_scores=1,
                                        use_node_id_as_component_id=1
                                        )
        else:
            components = nodes
        # print("%.3fs" % (time.time() - start))

        seg_local2super = np.array([nodes, components])
        # print("Storing seg_local2super LUT for threshold %.3f..." % threshold)
        lut_file = os.path.join(lut_dir_subseg, 'seg_local2super', str(block.block_id))
        np.savez_compressed(lut_file, seg=seg_local2super)

        nodes_super = np.unique(components)
        # print("Storing nodes_super LUT for threshold %.3f..." % threshold)
        lut_file = os.path.join(lut_dir_subseg, 'nodes_super', str(block.block_id))
        np.savez_compressed(lut_file, nodes=nodes_super)

        # filter for only outward edges
        nodes_in_vol = set(nodes)
        def not_in_graph(u, v):
            return u not in nodes_in_vol or v not in nodes_in_vol
        outward_edges = np.array([not_in_graph(n[0], n[1]) for n in edges])
        if len(outward_edges):
            edges = edges[outward_edges]

        local2super = {n: k for n, k in np.dstack((seg_local2super[0], seg_local2super[1]))[0]}
        for i in range(len(edges)):
            if edges[i][0] in local2super:
                if edges[i][0] != local2super[edges[i][0]]:
                    edges[i][0] = local2super[edges[i][0]]
            if edges[i][1] in local2super:
                if edges[i][1] != local2super[edges[i][1]]:
                    edges[i][1] = local2super[edges[i][1]]
        if len(edges):
            # np.unique doesn't work on empty arrays
            edges = np.unique(edges, axis=0)

        # print("Storing edges_super2local LUT for threshold %.3f..." % threshold)
        lut_file = os.path.join(lut_dir_subseg, 'edges_super2local', str(block.block_id))
        np.savez_compressed(lut_file, edges=edges)


if __name__ == "__main__":

    if sys.argv[1] == 'run':
        assert False, "Not tested"

    else:

        print(sys.argv)
        config_file = sys.argv[1]
        with open(config_file, 'r') as f:
            run_config = json.load(f)

        for key in run_config:
            globals()['%s' % key] = run_config[key]

        if run_config.get('block_id_add_one_fix', False):
            daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

        print("WORKER: Running with context %s" % os.environ['DAISY_CONTEXT'])
        client_scheduler = daisy.Client()

        db_client = pymongo.MongoClient(db_host)
        db = db_client[db_name]
        completion_db = db[completion_db_name]
        total_roi = daisy.Roi(total_roi_offset, total_roi_shape)

        while True:
            block = client_scheduler.acquire_block()
            if block is None:
                break

            get_chunkwise_lut(
                block,
                super_block_size,
                total_roi,
                lut_dir_local,
                lut_dir_subseg,
                merge_function,
                thresholds,
                super_chunk_size
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
