# import json
import logging
import sys
import os
import os.path as path

# import numpy as np

import daisy

import task_helper2 as task_helper
from task_04b_find_segment_blockwise import FindSegmentsBlockwiseTask2

logger = logging.getLogger(__name__)


class FindSegmentsBlockwiseTask3(task_helper.SlurmTask):

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    # out_file = daisy.Parameter()
    # out_dataset = daisy.Parameter()
    merge_function = daisy.Parameter()
    thresholds = daisy.Parameter()
    num_workers = daisy.Parameter()
    lut_dir = daisy.Parameter()

    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)

    block_size = daisy.Parameter([4000, 4096, 4096])

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        self.block_size = tuple(self.block_size)

        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))

        else:
            fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)
            total_roi = fragments.roi

        assert fragments.roi.contains(total_roi)

        self.out_dir = os.path.join(
            self.fragments_file,
            self.lut_dir)

        os.makedirs(self.out_dir, exist_ok=True)

        self.last_threshold = self.thresholds[-1]

        config = {
            'db_host': self.db_host,
            'db_name': self.db_name,
            'fragments_file': self.fragments_file,
            'lut_dir': self.lut_dir,
            'merge_function': self.merge_function,
            'thresholds': self.thresholds,
        }
        self.slurmSetup(
            config,
            '04c_find_segments_blockwise.py')

        check_function = self.block_done
        if self.overwrite:
            check_function = None

        self.schedule(
            total_roi,
            total_roi,
            total_roi,
            process_function=self.new_actor,
            check_function=check_function,
            num_workers=self.num_workers,
            read_write_conflict=False,
            fit='shrink',
            max_retries=100)

    def requires(self):
        if self.no_check_dependency:
            return []
        return [FindSegmentsBlockwiseTask2(global_config=self.global_config)]

    def block_done(self, block):

        block_id = block.block_id
        lookup = 'seg_local2global_%s_%d_single' % (
            self.merge_function, int(self.last_threshold*100))
        out_file = os.path.join(self.out_dir, lookup) + '.npz'
        logger.debug("Checking %s" % out_file)
        exists = path.exists(out_file)
        return exists


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    if global_config["Input"].get('block_id_add_one_fix', False):
        # fix for cb2_v4 dataset where one (1) was used for the first block id
        # future datasets should just use zero (0)
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

    daisy.distribute(
        [{'task': FindSegmentsBlockwiseTask3(global_config=global_config,
                                             **user_configs),
         'request': None}],
        global_config=global_config)
