# import json
# import os
import logging
import numpy as np
# import sys
import daisy
# import copy
from networkx import Graph
from scipy import ndimage

# from funlib.segment.arrays import relabel, replace_values
# from funlib.segment.arrays import replace_values
from segmentation_functions import agglomerate_in_block, segment, watershed_in_block

# debug
# import sys
# sys.path.insert(0, "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/lsd/")
# from lsd.parallel_fragments import watershed_in_block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("actor_myelin_convolution0")


def conv_pred(ndarray, z_step=True, xy_step=False):

    orig = ndarray.copy()
    if z_step:
        for i in range(len(ndarray)):
            if i != 0:
                ndarray[i] = np.minimum(orig[i-1], ndarray[i])
            if i != len(ndarray)-1:
                ndarray[i] = np.minimum(orig[i+1], ndarray[i])

    return ndarray


if __name__ == "__main__":

    file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/outputs/2019_04/cb2_synapse_cutout3_zoom4/cb2/130000/output.zarr"

    myelin_pred_ds = "volumes/myelin_affs_clean1"
    # initial_3d_myelin_segmentation_ds = "volumes/3d/hist_quant_50/myelin_segmentation_0.100"
    # myelin_affs_clean_ds = "volumes/myelin_affs_clean1"
    # myelin_boundary_grow_ds = "volumes/myelin_debug/myelin_boundary_grow"
    myelin_pred_conv_ds = "volumes/myelin_pred_conv0"

    myelin_pred_ds = daisy.open_ds(file, myelin_pred_ds)
    myelin_pred_ndarray = myelin_pred_ds.to_ndarray()

    total_roi = myelin_pred_ds.roi
    voxel_size = myelin_pred_ds.voxel_size
    block = daisy.block.Block(myelin_pred_ds.roi, myelin_pred_ds.roi, myelin_pred_ds.roi)
    # affs = convert_prediction_to_affs(block, myelin_pred_ds)

    z_steps = 1
    xy_steps = 5
    for i in range(z_steps):
        myelin_pred_ndarray = conv_pred(
            myelin_pred_ndarray, z_step=True, xy_step=False)

    myelin_pred_conv_ds = daisy.prepare_ds(
        file,
        myelin_pred_conv_ds,
        total_roi,
        voxel_size,
        dtype=myelin_pred_ds.dtype,
        compressor={'id': 'zlib', 'level': 5})
    myelin_pred_conv_ds[myelin_pred_conv_ds.roi] = myelin_pred_ndarray

    exit(0)

    # init
    print(myelin_pred_ds.dtype)
    assert myelin_pred_ds.dtype == np.uint8
    myelin_seg_ds = daisy.open_ds(file, initial_3d_myelin_segmentation_ds)
    myelin_seg = myelin_seg_ds[total_roi]
    print("Reading myelin_seg_ndarray...")
    myelin_seg_ndarray = myelin_seg.to_ndarray()

    myelin_affs_clean_ds = daisy.prepare_ds(
        file,
        myelin_affs_clean_ds,
        total_roi,
        voxel_size,
        dtype=myelin_pred_ds.dtype,
        # write_roi=block_roi_z_dim,
        compressor={'id': 'zlib', 'level': 5})
    assert myelin_affs_clean_ds.dtype == np.uint8

    myelin_boundary_grow_ds = daisy.prepare_ds(
        file,
        myelin_boundary_grow_ds,
        total_roi,
        voxel_size,
        np.uint64,
        # write_roi=block_roi_z_dim,
        compressor={'id': 'zlib', 'level': 5})

    # perform boundary grow of the segment
    # steps to perform boundary grow in pixels
    steps_3d = min(z_steps, xy_steps)
    steps_2d = max(z_steps, xy_steps) - steps_3d
    if steps_3d:
        grow_boundary(myelin_seg_ndarray, steps_3d, only_xy=False)
    if steps_2d:
        grow_boundary(myelin_seg_ndarray, steps_2d, only_xy=True)

    print("Writing myelin_boundary_grow_ds...")
    myelin_boundary_grow_ds[total_roi] = myelin_seg_ndarray

    # clean myelin affs using the boundary
    # myelin_non_boundary = np.logical_not(myelin_seg_ndarray)
    myelin_non_boundary = np.logical_and(myelin_seg_ndarray, 1)
    print("Reading myelin_seg_ndarray...")
    myelin_affs_ndarray = myelin_pred_ds[total_roi].to_ndarray()
    myelin_affs_ndarray[myelin_non_boundary] = 255
    print("Writing myelin_affs_clean_ds...")
    myelin_affs_clean_ds[total_roi] = myelin_affs_ndarray

    exit(0)

