# import json
import logging
import numpy as np
# import os
import sys
import daisy
import task_helper

from task_01_predict_blockwise import PredictTask
from task_predict_myelin import PredictMyelinTask

logger = logging.getLogger(__name__)


class MergeMyelinTask(task_helper.SlurmTask):
    '''.
    '''

    affs_file = daisy.Parameter()
    affs_dataset = daisy.Parameter()
    merged_affs_file = daisy.Parameter()
    merged_affs_dataset = daisy.Parameter()
    myelin_file = daisy.Parameter()
    myelin_dataset = daisy.Parameter()
    block_size = daisy.Parameter()
    num_workers = daisy.Parameter()
    downsample_xy = daisy.Parameter()

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        affs_ds = daisy.open_ds(self.affs_file, self.affs_dataset)
        # myelin_ds = daisy.open_ds(self.myelin_file, self.myelin_dataset)

        # print(affs_ds.roi)
        # print(myelin_ds.roi)
        # assert(affs_ds.roi == myelin_ds.roi)

        self.merged_affs_ds = daisy.prepare_ds(
            self.merged_affs_file,
            self.merged_affs_dataset,
            affs_ds.roi,
            affs_ds.voxel_size,
            np.uint8,
            write_roi=daisy.Roi((0, 0, 0), self.block_size),
            num_channels=3,
            compressor={'id': 'zlib', 'level': 5}
            )

        config = {
            'affs_file': self.affs_file,
            'affs_dataset': self.affs_dataset,
            'myelin_file': self.myelin_file,
            'myelin_dataset': self.myelin_dataset,
            'merged_affs_file': self.merged_affs_file,
            'merged_affs_dataset': self.merged_affs_dataset,
            'downsample_xy': self.downsample_xy,
        }

        self.slurmSetup(
            config, 'segway.myelin_scripts.actor_myelin_merge', python_module=True
            )

        # align read_roi to downsampled pixels
        ds_voxel_size = [n for n in affs_ds.voxel_size]
        ds_voxel_size[1] *= self.downsample_xy
        ds_voxel_size[2] *= self.downsample_xy
        print(ds_voxel_size)
        read_roi_begin = tuple([
            affs_ds.roi.get_begin()[0],
            int(affs_ds.roi.get_begin()[1] / ds_voxel_size[1]) * ds_voxel_size[1],
            int(affs_ds.roi.get_begin()[2] / ds_voxel_size[2]) * ds_voxel_size[2],
            ])
        read_roi_end = tuple([
            affs_ds.roi.get_end()[0],
            (int((affs_ds.roi.get_end()[1] - 1) / ds_voxel_size[1]) + 1) * ds_voxel_size[1],
            (int((affs_ds.roi.get_end()[2] - 1) / ds_voxel_size[2]) + 1) * ds_voxel_size[2],
            ])

        read_roi_shape = tuple([n-m for n, m in zip(read_roi_end, read_roi_begin)])
        read_roi = daisy.Roi(read_roi_begin, read_roi_shape)
        print(read_roi)

        for i in range(len(affs_ds.voxel_size)):
            assert read_roi_begin[i] % affs_ds.voxel_size[i] == 0
            assert read_roi_begin[i] % ds_voxel_size[i] == 0
            assert read_roi_end[i] % affs_ds.voxel_size[i] == 0
            assert read_roi_end[i] % ds_voxel_size[i] == 0

        self.schedule(
            total_roi=read_roi,
            read_roi=read_roi,
            write_roi=affs_ds.roi,
            process_function=self.new_actor,
            check_function=(self.check_block, lambda b: True),
            read_write_conflict=False,
            fit='shrink',
            num_workers=self.num_workers)

    def check_block(self, block):
        # logger.info("Checking if block %s is complete..." % block.write_roi)

        write_roi = self.merged_affs_ds.roi.intersect(block.write_roi)
        if write_roi.empty():
            logger.debug("Block outside of output ROI")
            return True

        s = 0
        quarter = (write_roi.get_end() - write_roi.get_begin()) / 4
        s += np.sum(self.merged_affs_ds[write_roi.get_begin() + quarter*1])
        s += np.sum(self.merged_affs_ds[write_roi.get_begin() + quarter*2])
        s += np.sum(self.merged_affs_ds[write_roi.get_begin() + quarter*3])
        logger.debug("Sum of center values in %s is %f" % (write_roi, s))

        return s != 0

    def requires(self):
        return [
            PredictTask(global_config=self.global_config),
            PredictMyelinTask(global_config=self.global_config)
            ]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
        [{'task': MergeMyelinTask(global_config=global_config,
                                  **user_configs),
         'request': None}],
        global_config=global_config)
