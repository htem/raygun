import daisy
import os
import json
import logging
from funlib.segment.arrays import replace_values
import sys
import time
import numpy as np
import pymongo

from task_04d_find_segment_blockwise import enumerate_blocks_in_chunks
import task_helper

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)
# np.set_printoptions(threshold=sys.maxsize, formatter={'all':lambda x: str(x)})


def remap_in_block(
        block,
        block_size,
        total_roi,
        lut_dir,
        merge_function,
        threshold,
        global_lut=None,
        chunk_size=None):
    '''Remap local subsegment IDs to global segment IDs using the global
    connected component LUTs computed in step 04c.
    This step is only necessary for computing whole-dataset segmentation
    (as opposed to a block-wise segmentation).'''

    logging.info("Received block %s" % block)

    if global_lut is None:
        global_lut = load_global_lut(threshold, lut_dir, merge_function)

    blocks = enumerate_blocks_in_chunks(
        block, block_size, chunk_size, total_roi)

    local_nodes_list = []
    for b in blocks:

        nodes_file = 'nodes_%s_%d/%d.npz' % (
            merge_function, int(threshold*100), b.block_id)
        nodes_file = os.path.join(lut_dir, nodes_file)
        logging.info("Loading nodes %s" % nodes_file)
        local_nodes_list.append(np.load(nodes_file)['nodes'])

    lens = [len(l) for l in local_nodes_list]
    chunk_start_index = [sum(lens[0:i+1]) for i in range(len(lens))]
    chunk_start_index.insert(0, 0)

    local_nodes_chunks = np.concatenate(local_nodes_list)

    logging.info("Remapping nodes %s" % nodes_file)
    start = time.time()
    remapped_chunks = replace_values(local_nodes_chunks, global_lut[0], global_lut[1])
    print("%.3fs" % (time.time() - start))

    for i, b in enumerate(blocks):

        local_nodes = local_nodes_chunks[chunk_start_index[i]:chunk_start_index[i+1]]
        remapped = remapped_chunks[chunk_start_index[i]:chunk_start_index[i+1]]

        # remove self-referencing entries
        non_self_refs = local_nodes != remapped
        local_nodes = local_nodes[non_self_refs]
        remapped = remapped[non_self_refs]

        lut = np.array([local_nodes, remapped])

        # write
        out_file = 'seg_local2global_%s_%d/%d.npz' % (
            merge_function, int(threshold*100), b.block_id)
        out_file = os.path.join(lut_dir, out_file)
        np.savez_compressed(out_file, fragment_segment_lut=lut)


def load_global_lut(threshold, lut_dir, merge_function, lut_filename=None):

    if lut_filename is None:
        lut_filename = 'seg_local2global_%s_%d_single' % (merge_function, int(threshold*100))
        # lut_filename = lut_filename + '_' + str(int(threshold*100))
    lut = os.path.join(
            lut_dir,
            lut_filename + '.npz')
    assert os.path.exists(lut), "%s does not exist" % lut
    start = time.time()
    logging.info("Reading global LUT...")
    lut = np.load(lut)['fragment_segment_lut']
    logging.info("%.3fs"%(time.time() - start))
    logging.info("Found %d fragments"%len(lut[0]))
    return lut


if __name__ == "__main__":

    if sys.argv[1] == 'run':

        assert False, "Not tested"

        user_configs, global_config = task_helper.parseConfigs(sys.argv[2:])
        config = user_configs
        config.update(global_config["ExtractSegmentationFromLUTBlockwiseTask"])

        find_segments(**config)

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
        block_size = daisy.Coordinate(block_size)

        lut_dir = os.path.join(
            fragments_file,
            lut_dir)

        global_luts = {}
        for threshold in thresholds:
            global_luts[threshold] = load_global_lut(threshold, lut_dir, merge_function)

        # global_lut = load_global_lut(threshold, lut_dir, merge_function)

        while True:
            block = client_scheduler.acquire_block()
            if block is None:
                break

            for threshold in thresholds:

                remap_in_block(
                    block,
                    block_size,
                    total_roi,
                    lut_dir,
                    merge_function,
                    threshold,
                    global_lut=global_luts[threshold],
                    chunk_size=tuple(chunk_size))

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
