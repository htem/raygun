# import json
import logging
import sys
import os
import os.path as path
from itertools import product
import numpy as np

import daisy
import task_helper2 as task_helper
from task_04c_find_segment_blockwise import FindSegmentsBlockwiseTask3

logger = logging.getLogger(__name__)


class FindSegmentsBlockwiseTask4(task_helper.SlurmTask):

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
    chunk_size = daisy.Parameter([2, 2, 2])

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        self.block_size = daisy.Coordinate(self.block_size) * tuple(self.chunk_size)

        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
        else:
            total_roi = fragments.roi
        self.total_roi = total_roi

        assert fragments.roi.contains(total_roi)
        
        read_roi = daisy.Roi((0,)*total_roi.dims(), self.block_size)
        write_roi = read_roi

        self.out_dir = os.path.join(
            self.fragments_file,
            self.lut_dir)

        os.makedirs(self.out_dir, exist_ok=True)

        for threshold in self.thresholds:
            os.makedirs(os.path.join(
                    self.out_dir,
                    "seg_local2global_%s_%d" % (self.merge_function, int(threshold*100)),
                    ),
                exist_ok=True)

        self.last_threshold = self.thresholds[-1]

        config = {
            'db_host': self.db_host,
            'db_name': self.db_name,
            'fragments_file': self.fragments_file,
            'lut_dir': self.lut_dir,
            'merge_function': self.merge_function,
            'thresholds': self.thresholds,
            'chunk_size': self.chunk_size,
            'block_size': self.block_size,
            'total_roi_offset': total_roi.get_offset(),
            'total_roi_shape': total_roi.get_shape(),
        }
        self.slurmSetup(
            config,
            '04d_find_segments_blockwise.py')

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
            fit='shrink',
            max_retries=100)

    def requires(self):
        if self.no_check_dependency:
            return []
        return [FindSegmentsBlockwiseTask3(global_config=self.global_config)]

    def block_done(self, chunk):

        # get the last chunk in the chunk
        blocks = enumerate_blocks_in_chunks(
            chunk, self.block_size, self.chunk_size, self.total_roi)
        block = blocks[-1]

        block_id = block.block_id
        lookup = 'seg_local2global_%s_%d/%d' % (
            self.merge_function,
            int(self.last_threshold*100),
            block_id
            )
        out_file = os.path.join(self.out_dir, lookup) + '.npz'
        exists = path.exists(out_file)
        if not exists:
            logger.info("%s not found" % out_file)
        return exists


def enumerate_blocks_in_chunks(block, block_size, chunk_size, total_roi):

    if chunk_size is None:
        return block

    block_size = daisy.Coordinate(block_size)
    chunk_size = daisy.Coordinate(chunk_size)

    blocks = []
    # roi_shape = block.requested_write_roi.get_shape()
    chunk_shape = block_size / chunk_size
    # print(roi_shape / chunk_shape)
    # print(product(*list(roi_shape / chunk_shape)))
    ref_roi = daisy.Roi(block.write_roi.get_offset(), chunk_shape)

    offsets = [range(n) for n in chunk_size]

    for offset_mult in product(*offsets):

        # print(offset_mult)
        shifted_roi = ref_roi.shift(chunk_shape*offset_mult)
        if total_roi.intersects(shifted_roi):
            blocks.append(
                daisy.Block(total_roi, shifted_roi, shifted_roi))

    return blocks

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    if global_config["Input"].get('block_id_add_one_fix', False):
        # fix for cb2_v4 dataset where one (1) was used for the first block id
        # future datasets should just use zero (0)
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

    daisy.distribute(
        [{'task': FindSegmentsBlockwiseTask4(global_config=global_config,
                                             **user_configs),
         'request': None}],
        global_config=global_config)
