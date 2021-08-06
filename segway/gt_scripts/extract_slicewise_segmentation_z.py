# import json
import logging
import sys
import daisy
import numpy as np
from networkx import Graph
import gt_tools
# sys.path.insert(0, "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation")

# from segway import task_helper
from segway.tasks.segmentation_functions import agglomerate_in_block, segment

logger = logging.getLogger(__name__)

'''
Change logs:

5/8/19: add task_config_files in config to automatically get the database address and database name

5/23/19:
- add option to use myelin merged_affs instead

'''


def process_block(
        block,
        affs_ds,
        fragments_ds,
        # output_file,
        thresholds,
        segmentation_dss,
        ):

    total_roi = block.write_roi
    rag = Graph()
    agglomerate_in_block(
        affs_ds,
        fragments_ds,
        total_roi,
        rag
        )

    segment(
        fragments_ds,
        roi=total_roi,
        rag=rag,
        thresholds=thresholds,
        segmentation_dss=segmentation_dss
        )


def check_block(block, ds):

    # ds = daisy.open_ds(self.out_file, self.out_dataset)
    # ds = self.ds
    write_roi = ds.roi.intersect(block.write_roi)
    if write_roi.empty():
        # logger.debug("Block outside of output ROI")
        return True

    center_coord = (write_roi.get_begin() +
                    write_roi.get_end()) / 2
    center_values = ds[center_coord]
    s = np.sum(center_values)
    # logger.debug("Sum of center values in %s is %f" % (write_roi, s))

    return s != 0


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    config = gt_tools.load_config(sys.argv[1])
    file = config["file"]

    if "myelin_mask_file" in config:
        affs_ds = daisy.open_ds(file, "volumes/merged_affs")
    else:
        affs_ds = daisy.open_ds(file, "volumes/affs")
    fragments_ds = daisy.open_ds(file, "volumes/fragments")

    num_workers = 12

    voxel_size = affs_ds.voxel_size

    total_roi_shape = affs_ds.roi.get_shape()
    block_roi_x_dim = [x for x in total_roi_shape]
    block_roi_x_dim[2] = voxel_size[2]
    block_roi_x_dim = daisy.Roi((0, 0, 0), block_roi_x_dim)
    block_roi_y_dim = [x for x in total_roi_shape]
    block_roi_y_dim[1] = voxel_size[1]
    block_roi_y_dim = daisy.Roi((0, 0, 0), block_roi_y_dim)
    block_roi_z_dim = [x for x in total_roi_shape]
    block_roi_z_dim[0] = voxel_size[0]
    block_roi_z_dim = daisy.Roi((0, 0, 0), block_roi_z_dim)

    # thresholds = [.1, .15, .85, .9]
    thresholds = [.1, .9]
    segmentation_dss = []

    for threshold in thresholds:

        segmentation_ds = daisy.prepare_ds(
            file,
            "volumes/segmentation_slice_z" + "_%.3f" % threshold,
            fragments_ds.roi,
            fragments_ds.voxel_size,
            fragments_ds.data.dtype,
            write_roi=block_roi_z_dim,
            compressor={'id': 'zlib', 'level': 5})

        segmentation_dss.append(segmentation_ds)

    # process block-wise
    daisy.run_blockwise(
        affs_ds.roi,
        block_roi_z_dim,
        block_roi_z_dim,
        process_function=lambda b: process_block(
            b,
            affs_ds,
            fragments_ds,
            thresholds,
            segmentation_dss
            ),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='valid')
