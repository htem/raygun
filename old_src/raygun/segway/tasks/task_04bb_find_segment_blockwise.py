# import json
import logging
import sys
import os
import os.path as path

# import numpy as np

import daisy

import task_helper2 as task_helper
from task_04ba_find_segment_blockwise import FindSegmentsBlockwiseTask2a

logger = logging.getLogger(__name__)


class FindSegmentsBlockwiseTask2b(task_helper.SlurmTask):

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

    block_size = daisy.Parameter()
    super_chunk_size = daisy.Parameter()
    block_id_add_one_fix = daisy.Parameter(False)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        self.block_size = tuple(self.block_size)
        super_block_size = (
            daisy.Coordinate(self.block_size) * tuple(self.super_chunk_size))

        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
            assert fragments.roi.contains(total_roi)
            total_roi = total_roi.grow(super_block_size, super_block_size)

        else:
            total_roi = fragments.roi.grow(super_block_size, super_block_size)

        read_roi = daisy.Roi((0,)*total_roi.dims(),
                             super_block_size).grow(super_block_size, super_block_size)
        write_roi = daisy.Roi((0,)*total_roi.dims(), super_block_size)

        lut_dir_out = os.path.join(self.fragments_file, self.lut_dir)
        super_lut_dir = 'super_%dx%dx%d_%s' % (
            self.super_chunk_size[0], self.super_chunk_size[1], self.super_chunk_size[2],
            self.merge_function)
        lut_dir_out = os.path.join(lut_dir_out, super_lut_dir)

        for threshold in self.thresholds:
            threshold_dir = lut_dir_out + '_%d' % int(threshold*100)
            os.makedirs(os.path.join(threshold_dir, "edges_super2super"), exist_ok=True)

        self.last_threshold = self.thresholds[-1]
        self.lut_dir_out = lut_dir_out

        config = {
            'db_host': self.db_host,
            'db_name': self.db_name,
            'fragments_file': self.fragments_file,
            'lut_dir': lut_dir_out,
            # 'merge_function': self.merge_function,
            'total_roi_offset': total_roi.get_offset(),
            'total_roi_shape': total_roi.get_shape(),
            # 'num_workers': self.num_workers,
            'thresholds': self.thresholds,
            'block_size': super_block_size,
            'block_id_add_one_fix': self.block_id_add_one_fix,
        }
        self.slurmSetup(
            config,
            '04bb_find_segments_blockwise.py')

        check_function = self.block_done
        if self.overwrite:
            check_function = None

        self.schedule(
            total_roi,
            read_roi,
            write_roi,
            process_function=self.new_actor,
            check_function=check_function,
            num_workers=self.num_workers,
            read_write_conflict=False,
            max_retries=self.max_retries,
            fit='shrink')

    def requires(self):
        if self.no_check_dependency:
            return []
        return [FindSegmentsBlockwiseTask2a(global_config=self.global_config)]

    def block_done(self, block):

        lut_dir = self.lut_dir_out + '_%d' % int(self.last_threshold*100)
        lookup = 'edges_super2super/%d.npz' % block.block_id
        out_file = os.path.join(lut_dir, lookup)
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
        global_config["FindSegmentsBlockwiseTask2b"]['block_id_add_one_fix'] = True

    daisy.distribute(
        [{'task': FindSegmentsBlockwiseTask2b(global_config=global_config,
                                             **user_configs),
         'request': None}],
        global_config=global_config)
