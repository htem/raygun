import daisy
import os
import json
import logging
from funlib.segment.arrays import replace_values
import sys
import time
import numpy as np
import pymongo

# from funlib.segment.graphs.impl import connected_components

# import task_helper
from task_04d_find_segment_blockwise import enumerate_blocks_in_chunks

logging.basicConfig(level=logging.INFO)


def get_chunkwise_lut(
        block,
        block_size,
        total_roi,
        super_lut_dir,
        merge_function,
        threshold,
        ):

    super_lut_dir = super_lut_dir + '_%d' % int(threshold*100)
    lut_file = os.path.join(super_lut_dir, 'seg_local2super', str(block.block_id) + '.npz')
    lut = np.load(lut_file)['seg']

    return lut


def segment_in_block(
        block,
        fragments_file,
        lut_dir,
        super_lut_dir,
        merge_function,
        thresholds,
        segmentations,
        fragments,
        block_size,
        total_roi,
        super_chunk_size,
        ):

    logging.info("Received chunk %s" % block)

    chunkwise_luts = {}
    for threshold in thresholds:

        chunkwise_lut = get_chunkwise_lut(
            block,
            block_size,
            total_roi,
            super_lut_dir,
            merge_function,
            threshold,
            )

        chunkwise_luts[threshold] = chunkwise_lut

    blocks = enumerate_blocks_in_chunks(
        block, block_size, super_chunk_size, total_roi)

    for block in blocks:

        logging.debug("Processing block %s" % block)

        logging.debug("Copying fragments to memory...")
        start = time.time()
        roi = total_roi.intersect(block.write_roi)
        fragments_ndarray = fragments.to_ndarray(roi)
        logging.debug("%.3fs"%(time.time() - start))

        for threshold in thresholds:

            logging.debug("Load local LUT...")
            start = time.time()
            local_lut = 'seg_frags2local_%s_%d/%d' % (merge_function, int(threshold*100), block.block_id)
            local_lut = os.path.join(lut_dir, local_lut + ".npz")
            if not os.path.exists(local_lut):
                # no segment to relabel
                print("%s not found" % local_lut)
                continue
            local_lut = np.load(local_lut)['fragment_segment_lut']
            logging.debug("Found %d fragments" % len(local_lut[0]))
            logging.debug("%.3fs"%(time.time() - start))

            logging.debug("Relabelling fragments to local segments")
            start = time.time()
            relabelled = replace_values(fragments_ndarray, local_lut[0], local_lut[1], inplace=False)
            logging.debug("%.3fs"%(time.time() - start))

            logging.debug("Relabelling fragments to global segments")
            start = time.time()
            chunkwise_lut = chunkwise_luts[threshold]
            relabelled = replace_values(relabelled, chunkwise_lut[0], chunkwise_lut[1], inplace=True)
            logging.debug("%.3fs"%(time.time() - start))

            logging.debug("Writing segments...")
            start = time.time()
            # print(relabelled)
            segmentation = segmentations[threshold]
            segmentation[roi] = relabelled
            logging.debug("%.3fs"%(time.time() - start))


def load_global_lut(threshold, lut_dir, lut_filename=None):

    if lut_filename is None:
        lut_filename = 'seg_local2global_%s_%d_single' % (merge_function, int(threshold*100))
    lut = os.path.join(
            lut_dir,
            lut_filename + '.npz')
    assert os.path.exists(lut), "%s does not exist" % lut
    start = time.time()
    logging.debug("Reading global LUT...")
    lut = np.load(lut)['fragment_segment_lut']
    logging.debug("%.3fs"%(time.time() - start))
    logging.debug("Found %d fragments"%len(lut[0]))
    return lut


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

        print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
        client_scheduler = daisy.Client()

        db_client = pymongo.MongoClient(db_host)
        db = db_client[db_name]
        completion_db = db[completion_db_name]

        total_roi = daisy.Roi(total_roi_offset, total_roi_shape)
        fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

        segmentations = {}
        for threshold in thresholds:
            ds = out_dataset + "_%.3f" % threshold
            segmentations[threshold] = daisy.open_ds(out_file, ds, mode='r+')

        while True:
            block = client_scheduler.acquire_block()
            if block is None:
                break

            segment_in_block(
                block,
                fragments_file,
                lut_dir,
                super_lut_dir,
                merge_function,
                thresholds,
                segmentations,
                fragments,
                block_size,
                total_roi,
                super_chunk_size,
                # write_lut_only
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
