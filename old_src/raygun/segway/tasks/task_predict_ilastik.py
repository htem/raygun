# import json
import logging
import numpy as np
# import os
import sys
import daisy
import task_helper2 as task_helper

logger = logging.getLogger(__name__)


class PredictIlastikTask(task_helper.SlurmTask):
    '''.
    '''

    raw_file = daisy.Parameter()
    raw_dataset = daisy.Parameter()

    roi_offset = daisy.Parameter(None)
    roi_shape = daisy.Parameter(None)
    # sub_roi is used to specify the region of interest while still allocating
    # the entire input raw volume. It is useful when there is a chance that
    # sub_roi will be increased in the future.
    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)

    context = daisy.Parameter()
    block_size = daisy.Parameter()

    num_workers = daisy.Parameter()
    out_file = daisy.Parameter()
    out_dataset = daisy.Parameter()
    # downsample_xy = daisy.Parameter()
    network_voxel_size = daisy.Parameter()
    lazyflow_num_threads = daisy.Parameter()
    lazyflow_mem = daisy.Parameter()
    ilastik_project_path = daisy.Parameter()

    clamp_threshold = daisy.Parameter(0)
    replace_section_list = daisy.Parameter([])
    block_id_add_one_fix = daisy.Parameter(False)

    # user provided myelin gt
    user_gt = daisy.Parameter(None)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        self.context = daisy.Coordinate(self.context)

        raw_ds = daisy.open_ds(self.raw_file, self.raw_dataset)

        sched_roi, dataset_roi = task_helper.compute_compatible_roi(
            roi_offset=self.roi_offset,
            roi_shape=self.roi_shape,
            sub_roi_offset=self.sub_roi_offset,
            sub_roi_shape=self.sub_roi_shape,
            roi_context=self.context,
            source_roi=raw_ds.roi,
            chunk_size=self.block_size,
            )

        read_roi = daisy.Roi((0,)*sched_roi.dims(),
                             self.block_size).grow(self.context, self.context)
        write_roi = daisy.Roi((0,)*sched_roi.dims(), self.block_size)

        self.ilastik_out = daisy.prepare_ds(
            self.out_file,
            self.out_dataset,
            dataset_roi,
            self.network_voxel_size,
            np.uint8,
            daisy.Roi((0, 0, 0), self.block_size),
            compressor={'id': 'zlib', 'level': 5}
            )
        assert self.ilastik_out.data.dtype == np.uint8

        if self.user_gt is not None:
            assert False, "Untested"
            # if gt is provided, use it and don't run ilastik
            # only supporting h5 exports from ilastik for now
            assert self.user_gt.endswith(".h5")
            gt = daisy.open_ds(self.user_gt, "exported_data")
            logger.info("gt.voxel_size: %s", gt.voxel_size)
            logger.info("gt.shape: %s", gt.shape)
            logger.info("network_voxel_size: %s", self.network_voxel_size)
            gt_ndarray = gt.to_ndarray()
            logger.info("gt_ndarray.shape: %s", gt_ndarray.shape)
            gt_ndarray = gt_ndarray[:, :, :, 0]
            np.place(gt_ndarray, gt_ndarray == 1, 0)
            np.place(gt_ndarray, gt_ndarray == 2, 255)
            gt_array = daisy.Array(gt_ndarray,
                                   roi=sched_roi,
                                   voxel_size=self.network_voxel_size,
                                   )
            self.ilastik_out[sched_roi] = gt_array[sched_roi].to_ndarray()

        config = {
            'raw_file': self.raw_file,
            'raw_dataset': self.raw_dataset,
            'block_size': self.block_size,
            'out_file': self.out_file,
            'out_dataset': self.out_dataset,
            'network_voxel_size': self.network_voxel_size,
            'lazyflow_num_threads': self.lazyflow_num_threads,
            'lazyflow_mem': self.lazyflow_mem,
            'ilastik_project_path': self.ilastik_project_path,
            'clamp_threshold': self.clamp_threshold,
            'replace_section_list': self.replace_section_list,
            'block_id_add_one_fix': self.block_id_add_one_fix,
        }

        self.slurmSetup(
            config, 'actor_predict_ilastik.py',
            python_interpreter='/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/ilastik-1.3.2post1-Linux/bin/python3.6',
            )

        check_function = (self.check_block, lambda b: True)
        if self.overwrite:
            check_function = None
        if self.no_precheck:
            check_function = None

        self.schedule(
            total_roi=sched_roi,
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

        write_roi = self.ilastik_out.roi.intersect(block.write_roi)
        if write_roi.empty():
            logger.debug("Block outside of output ROI")
            return True

        s = 0
        quarter = (write_roi.get_end() - write_roi.get_begin()) / 4
        s += self.ilastik_out[write_roi.get_begin() + quarter*1]
        s += self.ilastik_out[write_roi.get_begin() + quarter*2]
        s += self.ilastik_out[write_roi.get_begin() + quarter*3]
        logger.debug("Sum of center values in %s is %f" % (write_roi, s))

        return s != 0


if __name__ == "__main__":

    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
        [{'task': PredictIlastikTask(global_config=global_config,
                                      **user_configs),
         'request': None}],
        global_config=global_config)
