import daisy
import os
import json
import logging
from funlib.segment.arrays import replace_values
import sys
import time
import numpy as np
import pymongo

import task_helper

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)


def segment_in_block(
        block,
        lut_dir,
        merge_function,
        thresholds,
        segmentations,
        fragments):

    logging.info("Received block %s" % block)

    logging.info("Copying fragments to memory...")
    start = time.time()
    fragments = fragments.to_ndarray(block.write_roi)
    logging.info("%.3fs"%(time.time() - start))

    for threshold in thresholds:

        segmentation = segmentations[threshold]

        logging.info("Load local LUT...")
        start = time.time()
        local_lut = 'seg_frags2local_%s_%d/%d' % (merge_function, int(threshold*100), block.block_id)
        local_lut = np.load(os.path.join(lut_dir, local_lut + ".npz"))['fragment_segment_lut']
        logging.info("Found %d fragments" % len(local_lut[0]))
        logging.info("%.3fs"%(time.time() - start))

        logging.info("Relabelling fragments to local segments")
        start = time.time()
        relabelled = replace_values(fragments, local_lut[0], local_lut[1], inplace=False)
        logging.info("%.3fs"%(time.time() - start))

        logging.info("Load global LUT...")
        start = time.time()
        global_lut = 'seg_local2global_%s_%d/%d' % (merge_function, int(threshold*100), block.block_id)
        global_lut = np.load(os.path.join(lut_dir, global_lut + ".npz"))['fragment_segment_lut']
        logging.info("Found %d fragments" % len(global_lut[0]))
        logging.info("%.3fs"%(time.time() - start))

        logging.info("Relabelling fragments to global segments")
        start = time.time()
        relabelled = replace_values(relabelled, global_lut[0], global_lut[1], inplace=True)
        logging.info("%.3fs"%(time.time() - start))

        logging.info("Writing segments...")
        start = time.time()
        segmentation[block.write_roi] = relabelled
        logging.info("%.3fs"%(time.time() - start))


def extract_segmentation(
        fragments_file,
        fragments_dataset,
        lut_filename,
        lut_dir,
        merge_function,
        threshold,
        out_file,
        out_dataset,
        num_workers,
        roi_offset=None,
        roi_shape=None,
        run_type=None,
        **kwargs):

    lut_dir = os.path.join(
        fragments_file,
        lut_dir)

    segment_in_block(
        roi_offset,
        roi_shape,
        lut_dir,
        merge_function,
        threshold,
        segmentation,
        fragments,
        lut)


def load_global_lut(threshold, lut_dir, lut_filename=None):

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

        # total_roi = daisy.Roi(total_roi_offset, total_roi_shape)

        lut_dir = os.path.join(
            fragments_file,
            lut_dir)
        # global_lut = load_global_lut(threshold, lut_dir)
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
                lut_dir,
                merge_function,
                thresholds,
                segmentations,
                fragments)

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
