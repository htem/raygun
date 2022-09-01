import json
import logging
import numpy as np
# import os
import sys
# import datetime
# import multiprocessing
# from PIL import Image

import daisy
import task_helper

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class RenderRawToZarrFromTiffTask(task_helper.SlurmTask):
    '''
    '''

    # raw_file = daisy.Parameter()
    # raw_dataset = daisy.Parameter()
    # out_file = daisy.Parameter()
    # out_dataset = daisy.Parameter()
    # block_size_in_chunks = daisy.Parameter(None)
    # predict_file = daisy.Parameter(None)

    # sub_roi is used to specify the region of interest while still allocating
    # the entire input raw volume. It is useful when there is a chance that
    # sub_roi will be increased in the future.
    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)
    sub_roi_begin = daisy.Parameter(None)
    sub_roi_end = daisy.Parameter(None)

    num_workers = daisy.Parameter(4)

    # should be the chunk size of the ZARR vol
    # check .zarray
    block_size = daisy.Parameter([25*40, 2048*4, 2048*4])
    max_retries = daisy.Parameter(2)

    zarr_file = daisy.Parameter()
    zarr_dataset = daisy.Parameter()
    voxel_size = daisy.Parameter([40, 4, 4])
    aligned_dir_path = daisy.Parameter("/n/groups/htem/temcagt/datasets/cb2/intersection/aligned_links/")
    y_folder_block_size = daisy.Parameter(8192)
    y_tile_block_size = daisy.Parameter(2048)
    x_tile_block_size = daisy.Parameter(2048)

    bad_sections = daisy.Parameter([
        73, 81, 88, 91, 135, 143,
        175, 184, 186, 187, 192, 193, 194,
        202, 217, 276,
        340, 341, 366, 393,
        387,  # to fix, not parallel rendered
        445, 448, 457, 493, 496, 497,
        509, 512, 538, 543, 548, 558, 561, 565, 566, 567, 568,
        # 569,  # to fix # fixed, but misaligned
        572, 579, 580, 586, 594, 604, 610, 621,
        622,  # to fix
        623, 627, 672, 674, 678, 683, 685,
        688,  # to fix, tiff rendered?
        689, 719, 724, 727, 730, 745, 746, 748, 771, 777, 786, 791, 797, 799,
        801,
        802,  # to ref
        803, 804, 811, 812, 825, 827, 828, 832, 836,
        844, 849, 851, 853, 854, 881, 886, 887, 888,
        891, 893, 895, 899,
        905, 906, 917, 925, 927, 941, 950, 960, 961,
        964, 970, 977, 991, 997,
        1010, 1012, 1014, 1016, 1023, 1027,
        1028,  # to fix
        1032, 1040, 1044, 1047, 1052, 1062, 1063,
        1067, 1068, 1071, 1072,
        1073,  # to fix
        1076, 1077, 1082, 1093, 1098,
        1100, 1112,
        1113,  # to ref
        1114, 1115,
        1116,  # to ref
        1120, 1163, 1164
    ])

    # to run fix:
    # try: 507, 511, 521 533 540 554
    # 1100 - 1125 (also wrong with Logan's, test first)

    avail_sections = daisy.Parameter([])

    # 175: clearly misaligned
    # 303 misaligned? used to be okay

    # 48573, 142548, 334
    # 334: missing?
    # missing? 344, 346, 387, 388 393 398 486 493 

    # 688 broken soft link

    log_dir = daisy.Parameter(None)

    def prepare(self):

        self.raw = daisy.open_ds(self.zarr_file, self.zarr_dataset, 'r+')

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:

            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
            total_roi = total_roi.snap_to_grid(
                self.block_size,
                mode='grow')

        elif self.sub_roi_begin is not None and self.sub_roi_end is not None:

            shape = daisy.Coordinate(tuple(self.sub_roi_end)) - daisy.Coordinate(tuple(self.sub_roi_begin))
            total_roi = daisy.Roi(
                tuple(self.sub_roi_begin), shape)
            total_roi = total_roi.snap_to_grid(
                self.block_size,
                mode='grow')

        else:

            # assert False, "Untested"
            total_roi = self.raw.roi

            # assert total_roi == total_roi.snap_to_grid(
            #     self.block_size,
            #     mode='grow'), "Sanity check for block_size"

            total_roi = total_roi.snap_to_grid(
                self.block_size,
                mode='shrink')

        read_roi = daisy.Roi((0, 0, 0), self.block_size)
        write_roi = daisy.Roi((0, 0, 0), self.block_size)

        logger.info("Following sizes in world units:")
        logger.info("read_roi  = %s" % (read_roi,))
        logger.info("write_roi = %s" % (write_roi,))
        logger.info("total_roi = %s" % (total_roi,))

        # self.task_done = False  # hack because we're not calling slurmSetup
        # error_log_f = "task_fix_raw_from_catmaid.error_blocks.%s" % str(datetime.datetime.now()).replace(' ', '_')
        # precheck_log_f = "task_fix_raw_from_catmaid.precheck_blocks.%s" % str(datetime.datetime.now()).replace(' ', '_')

        # self.error_log = open(error_log_f, "w")
        # self.precheck_log = open(precheck_log_f, "w")
        # self.shared_precheck_blocks = multiprocessing.Manager().list()
        # self.shared_error_blocks = multiprocessing.Manager().list()

        # self.launch_process_cmd = ''

        # self.catmaid_tile_size_pixel = daisy.Coordinate(self.catmaid_tile_size) / self.raw.voxel_size

        # check_function = (lambda b: False,
        #                   lambda b: self.check_block(b, False))

        # add 0 to 70 for cb2
        for i in range(70):
            self.bad_sections.append(i)
        # add 1170 to 1250 for cb2
        for i in range(1170, 1300):
            self.bad_sections.append(i)

        config = {
            'zarr_file': self.zarr_file,
            'zarr_dataset': self.zarr_dataset,
            'voxel_size': self.voxel_size,
            'aligned_dir_path': self.aligned_dir_path,
            'y_folder_block_size': self.y_folder_block_size,
            'y_tile_block_size': self.y_tile_block_size,
            'x_tile_block_size': self.x_tile_block_size,
            'bad_sections': self.bad_sections,
            "avail_sections": self.avail_sections
        }

        self.slurmSetup(config,
                        "./actor_render_raw_to_zarr_from_tiff.py",
                        )

        check_function = (
                lambda b: self.check_block(b, precheck=True),
                lambda b: self.check_block(b, precheck=False)
                )

        if self.overwrite:
            check_function = None

        self.schedule(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=self.new_actor,
            check_function=check_function,
            read_write_conflict=False,
            fit='overhang',
            num_workers=self.num_workers,
            max_retries=self.max_retries)

    def check_block(self, block, precheck):
        logger.debug("Checking if block %s is complete..." % block.write_roi)

        write_roi = self.raw.roi.intersect(block.write_roi)
        if write_roi.empty():
            logger.debug("Block outside of output ROI")
            return True

        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            logger.debug("Skipping block with db check")
            return True

        s = 0
        quarter = (write_roi.get_end() - write_roi.get_begin()) / 4
        center = write_roi.get_begin() + quarter*2
        s += np.sum(self.raw[center])
        s += np.sum(self.raw[center - daisy.Coordinate((240, 24, 24))])
        s += np.sum(self.raw[center + daisy.Coordinate((240, 24, 24))])
        logger.debug("Sum of center values in %s is %f" % (write_roi, s))

        done = s != 0
        if done:
            self.recording_block_done(block)

        # TODO: this should be filtered by post check and not pre check
        # if (s == 0):
        #     self.log_error_block(block)

        return done


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    req_roi = None
    if "request_offset" in global_config["Input"]:
        req_roi = daisy.Roi(
            tuple(global_config["Input"]["request_offset"]),
            tuple(global_config["Input"]["request_shape"]))
        req_roi = [req_roi]

    daisy.distribute(
        [{'task': RenderRawToZarrFromTiffTask(global_config=global_config,
                                  **user_configs),
         'request': req_roi}],
        global_config=global_config)
