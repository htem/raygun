# import json
# import os
import logging
import numpy as np
import daisy

from myelin_functions import grow_boundary, segmentation_to_affs

# import copy
# from networkx import Graph
# from scipy import ndimage

# from funlib.segment.arrays import relabel, replace_values
# from funlib.segment.arrays import replace_values
# from segmentation_functions import agglomerate_in_block, segment, watershed_in_block

# debug
# import sys
# sys.path.insert(0, "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/lsd/")
# from lsd.parallel_fragments import watershed_in_block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("actor_myelin_make_affs_from_segments")


if __name__ == "__main__":

    file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/outputs/2019_04/cb2_synapse_cutout3_zoom4/cb2/130000/output.zarr"

    # myelin_pred_ds = "volumes/myelin_affs"
    initial_3d_myelin_segmentation_ds = "volumes/conv1/3d/hist_quant_75/myelin_segmentation_0.950"
    myelin_affs_ds = "volumes/myelin_affs_final"
    # myelin_boundary_grow_ds = "volumes/myelin_debug/myelin_boundary_grow"
    # affs_ds = "volumes/affs"
    z_steps = 0
    xy_steps = 2

    # init
    # myelin_pred_ds = daisy.open_ds(file, myelin_pred_ds)
    # print(myelin_pred_ds.dtype)
    # assert myelin_pred_ds.dtype == np.uint8
    myelin_seg_ds = daisy.open_ds(file, initial_3d_myelin_segmentation_ds)
    total_roi = myelin_seg_ds.roi
    voxel_size = myelin_seg_ds.voxel_size
    myelin_seg = myelin_seg_ds[total_roi]
    print("Reading myelin_seg_ndarray...")
    myelin_seg_ndarray = myelin_seg.to_ndarray()

    myelin_affs_ds = daisy.prepare_ds(
        file,
        myelin_affs_ds,
        total_roi,
        voxel_size,
        dtype=np.uint8,
        # dtype=np.float32,
        num_channels=3,
        # write_roi=block_roi_z_dim,
        compressor={'id': 'zlib', 'level': 5})
    assert myelin_affs_ds.dtype == np.uint8

    # perform boundary grow of the segment
    # steps to perform boundary grow in pixels
    steps_3d = min(z_steps, xy_steps)
    steps_2d = max(z_steps, xy_steps) - steps_3d
    if steps_3d:
        grow_boundary(myelin_seg_ndarray, steps_3d, only_xy=False)
    if steps_2d:
        grow_boundary(myelin_seg_ndarray, steps_2d, only_xy=True)

    affs = segmentation_to_affs(myelin_seg_ndarray)
    myelin_affs_ds[total_roi] = affs

    # # clean myelin affs using the boundary
    # # myelin_non_boundary = np.logical_not(myelin_seg_ndarray)
    # myelin_non_boundary = np.logical_and(myelin_seg_ndarray, 1, dtype=np.uint8)
    # print("Reading myelin_seg_ndarray...")
    # # myelin_affs_ndarray = myelin_pred_ds[total_roi].to_ndarray()
    # # myelin_affs_ndarray[myelin_non_boundary] = 0
    # # print("Writing myelin_affs_ds...")
    # # a = np.zeros_like(myelin_non_boundary.shape, dtype=np.uint8)
    # # a[myelin_non_boundary] = 255
    # # myelin_non_boundary *= 255
    # myelin_non_boundary = myelin_non_boundary * 255
    # myelin_affs_ds[total_roi] = myelin_non_boundary

    exit(0)

