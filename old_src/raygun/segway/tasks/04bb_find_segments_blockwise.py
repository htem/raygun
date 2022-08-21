import daisy
import json
import logging
import sys
# import time
import os

import pymongo
import numpy as np

import task_helper

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.persistence.shared_graph_provider').setLevel(logging.DEBUG)
# np.set_printoptions(threshold=sys.maxsize, formatter={'all':lambda x: str(x)})

logger = logging.getLogger(__name__)


def replace_fragment_ids(
        db_host,
        db_name,
        block,
        lut_dir,
        total_roi,
        thresholds,
        run_type=None,
        block_id=None,
        **kwargs):
    '''Compute subsegment2subsegment edges.
    Necessary for auto-grow in dahlia.'''

    base_lut_dir_out = lut_dir

    # debug_mode = False

    for threshold in thresholds:

        # if block.block_id == 2423 and int(threshold*100) == 50:
        #     debug_mode = True
        # else:
        #     debug_mode = False

        lut_dir_out = base_lut_dir_out + '_%d' % int(threshold*100)

        edge_lut_file = os.path.join(lut_dir_out, 'edges_super2local', str(block_id) + '.npz')
        edges = np.load(edge_lut_file)['edges']

        if len(edges) == 0:
            lut_file = os.path.join(lut_dir_out, 'edges_super2super', str(block_id))
            np.savez_compressed(lut_file, edges=edges)
            continue

        super_fragment_nodes = []
        node_lut_file = os.path.join(lut_dir_out, 'nodes_super', str(block.block_id) + '.npz')
        super_fragment_nodes.extend(np.load(node_lut_file)['nodes'])

        seg_lut_file = os.path.join(lut_dir_out, 'seg_local2super', str(block.block_id) + '.npz')
        lut = np.load(seg_lut_file)['seg']
        local2super = {n: k for n, k in np.dstack((lut[0], lut[1]))[0]}

        bilateral_edges = []

        adj_blocks = generate_adjacent_blocks(roi_offset, roi_shape, total_roi)
        for adj_block in adj_blocks:
            adj_block_id = adj_block.block_id

            node_lut_file = os.path.join(lut_dir_out, 'nodes_super', str(adj_block_id) + '.npz')
            super_fragment_nodes.extend(np.load(node_lut_file)['nodes'])

            adj_seg_lut_file = os.path.join(lut_dir_out, 'seg_local2super', str(adj_block_id) + '.npz')

            if not os.path.exists(adj_seg_lut_file):
                logging.info("Skipping %s.." % adj_seg_lut_file)
                assert False
                continue

            adj_lut = np.load(adj_seg_lut_file)['seg']
            adj_local2super = {n: k for n, k in np.dstack((adj_lut[0], adj_lut[1]))[0]}

            for i in range(len(edges)):
                if edges[i][0] in adj_local2super:
                    if edges[i][0] != adj_local2super[edges[i][0]]:
                        edges[i][0] = adj_local2super[edges[i][0]]
                if edges[i][1] in adj_local2super:
                    if edges[i][1] != adj_local2super[edges[i][1]]:
                        edges[i][1] = adj_local2super[edges[i][1]]

            # add bilaral edges to the current block
            adj_edge_lut_file = os.path.join(lut_dir_out, 'edges_super2local', str(adj_block_id) + '.npz')
            adj_edges = np.load(adj_edge_lut_file)['edges']
            for i in range(len(adj_edges)):
                edge = adj_edges[i]
                valid = False
                if edge[0] in local2super:
                    valid = True
                    edge[0] = local2super[edge[0]]
                if edge[1] in local2super:
                    valid = True
                    edge[1] = local2super[edge[1]]
                if valid:
                    bilateral_edges.append(edge)

        super_fragment_nodes = set(super_fragment_nodes)
        def valid_super_fragments(u, v):
            return u in super_fragment_nodes and v in super_fragment_nodes
        valid_super_edges = np.array([valid_super_fragments(n[0], n[1]) for n in edges])
        edges = edges[valid_super_edges]


        if len(bilateral_edges):
            edges = np.append(edges, bilateral_edges, axis=0)

        if len(edges):
            # np.unique doesn't work on empty arrays
            edges = np.unique(edges, axis=0)

        lut_file = os.path.join(lut_dir_out, 'edges_super2super', str(block_id))
        # if debug_mode:
        #     print("final edges:", edges)
        #     print("lut_file:", lut_file)
        #     print("block.block_id:", block.block_id)
        #     print("block_id:", block_id)
        #     pass
        #     # exit(0)
        np.savez_compressed(lut_file, edges=edges)


def generate_adjacent_blocks(roi_offset, roi_shape, total_roi):

    blocks = []
    current_block_roi = daisy.Roi(roi_offset, roi_shape)

    # print("total_roi:", total_roi)
    # print("current_block_roi:", current_block_roi)

    total_write_roi = total_roi.grow(-roi_shape, -roi_shape)
    # print("total_write_roi:", total_write_roi)

    for offset_mult in [
            (-1, 0, 0),
            (+1, 0, 0),
            (0, -1, 0),
            (0, +1, 0),
            (0, 0, -1),
            (0, 0, +1),
            ]:

        shifted_roi = current_block_roi.shift(roi_shape*offset_mult)
        # print("shifted_roi:", shifted_roi)
        if total_write_roi.intersects(shifted_roi):
            blocks.append(
                daisy.Block(total_roi, shifted_roi, shifted_roi))

    return blocks


if __name__ == "__main__":

    if sys.argv[1] == 'run':

        user_configs, global_config = task_helper.parseConfigs(sys.argv[2:])
        config = user_configs
        config.update(global_config["FindSegmentsBlockwiseTask2"])

        replace_fragment_ids(**config)

    else:

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

        total_roi = daisy.Roi(total_roi_offset, total_roi_shape)

        while True:
            block = client_scheduler.acquire_block()
            if block is None:
                break

            print(block)

            roi_offset = block.write_roi.get_offset()
            roi_shape = daisy.Coordinate(tuple(block_size))

            replace_fragment_ids(
                db_host=db_host,
                db_name=db_name,
                block=block,
                lut_dir=lut_dir,
                block_id=block.block_id,
                total_roi=total_roi,
                roi_offset=roi_offset,
                roi_shape=roi_shape,
                thresholds=thresholds,
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
