# import json
import logging
import numpy as np
# import os
import sys
import daisy
import task_helper

logger = logging.getLogger(__name__)


class PredictMyelinTask(task_helper.SlurmTask):
    '''.
    '''

    raw_file = daisy.Parameter()
    raw_dataset = daisy.Parameter()
    roi_offset = daisy.Parameter(None)
    roi_shape = daisy.Parameter(None)
    block_size = daisy.Parameter()
    num_workers = daisy.Parameter()
    myelin_file = daisy.Parameter()
    myelin_dataset = daisy.Parameter()
    downsample_xy = daisy.Parameter()
    lazyflow_num_threads = daisy.Parameter()
    lazyflow_mem = daisy.Parameter()
    ilastik_project_path = daisy.Parameter()

    # user provided myelin gt
    user_gt = daisy.Parameter(None)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        raw_ds = daisy.open_ds(self.raw_file, self.raw_dataset)

        myelin_voxel_size = [i for i in raw_ds.voxel_size]
        myelin_voxel_size[1] = myelin_voxel_size[1] * self.downsample_xy
        myelin_voxel_size[2] = myelin_voxel_size[2] * self.downsample_xy

        if self.roi_offset is None and self.roi_shape is None:
            total_roi = raw_ds.roi
        else:
            assert(self.roi_offset is not None)
            assert(self.roi_shape is not None)
            total_roi = daisy.Roi(
                tuple(self.roi_offset), tuple(self.roi_shape))

        self.myelin_out = daisy.prepare_ds(
            self.myelin_file,
            self.myelin_dataset,
            total_roi,
            myelin_voxel_size,
            np.uint8,
            daisy.Roi((0, 0, 0), self.block_size),
            compressor={'id': 'zlib', 'level': 5}
            )
        assert self.myelin_out.data.dtype == np.uint8

        if self.user_gt is not None:
            # if gt is provided, use it and don't run ilastik
            # only supporting h5 exports from ilastik for now
            assert self.user_gt.endswith(".h5")
            gt = daisy.open_ds(self.user_gt, "exported_data")
            logger.info("gt.voxel_size: %s", gt.voxel_size)
            logger.info("gt.shape: %s", gt.shape)
            logger.info("myelin_voxel_size: %s", myelin_voxel_size)
            gt_ndarray = gt.to_ndarray()
            logger.info("gt_ndarray.shape: %s", gt_ndarray.shape)
            gt_ndarray = gt_ndarray[:, :, :, 0]
            np.place(gt_ndarray, gt_ndarray == 1, 0)
            np.place(gt_ndarray, gt_ndarray == 2, 255)
            # print(gt_ndarray)
            # logger.debug("n0.shape: ", n0.shape)
            gt_array = daisy.Array(gt_ndarray,
                                   roi=total_roi,
                                   voxel_size=myelin_voxel_size,
                                   # voxel_size=[40, 8, 8],
                                   )
            # print(gt_array[total_roi].to_ndarray())
            self.myelin_out[total_roi] = gt_array[total_roi].to_ndarray()

        read_roi = total_roi
        write_roi = total_roi

        config = {
            'raw_file': self.raw_file,
            'raw_dataset': self.raw_dataset,
            'block_size': self.block_size,
            'myelin_file': self.myelin_file,
            'myelin_dataset': self.myelin_dataset,
            'downsample_xy': self.downsample_xy,
            'lazyflow_num_threads': self.lazyflow_num_threads,
            'lazyflow_mem': self.lazyflow_mem,
            'ilastik_project_path': self.ilastik_project_path,
        }

        self.slurmSetup(
            config, '../myelin_scripts/actor_myelin_prediction.py',
            # python_interpreter='/home/tmn7/programming/ilastik-1.3.2post1-Linux/bin/python3.6',
            python_interpreter='/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/ilastik-1.3.2post1-Linux/bin/python3.6',
            )

        check_function = (self.check_block, lambda b: True)
        if self.no_precheck:
            check_function = None

        self.schedule(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=self.new_actor,
            check_function=check_function,
            read_write_conflict=False,
            fit='shrink',
            num_workers=self.num_workers)

    def check_block(self, block):

        if self.user_gt is not None:
            return True

        write_roi = self.myelin_out.roi.intersect(block.write_roi)
        if write_roi.empty():
            logger.debug("Block outside of output ROI")
            return True

        s = 0
        quarter = (write_roi.get_end() - write_roi.get_begin()) / 4
        s += self.myelin_out[write_roi.get_begin() + quarter*1]
        s += self.myelin_out[write_roi.get_begin() + quarter*2]
        s += self.myelin_out[write_roi.get_begin() + quarter*3]
        logger.debug("Sum of center values in %s is %f" % (write_roi, s))

        return s != 0


if __name__ == "__main__":

    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
        [{'task': PredictMyelinTask(global_config=global_config,
                                      **user_configs),
         'request': None}],
        global_config=global_config)

    # configs = {}
    # for config in sys.argv[1:]:
    #     with open(config, 'r') as f:
    #         configs = {**json.load(f), **configs}
    # aggregateConfigs(configs)
    # print(configs)

    # daisy.distribute([{'task': PredictMyelinTask(), 'request': None}],
    #                  global_config=global_config)
