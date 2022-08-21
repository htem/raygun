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
logger = logging.getLogger("actor_myelin_segmentation")


def convert_prediction_to_affs(
        block,
        myelin_ds,
        agglomerate_in_xy
        # myelin_affs_ds
        ):

    # assuming that myelin prediction is downsampled by an integer
    # myelin_affs_array = myelin_affs_ds[block.write_roi].to_ndarray()
    myelin_array = myelin_ds[block.write_roi].to_ndarray()
    affs_shape = tuple([3] + list(myelin_array.shape))
    myelin_affs_array = np.zeros(affs_shape, dtype=myelin_array.dtype)

    # thresholding
    low_threshold = 128
    np.place(myelin_array, myelin_array < low_threshold, [0])

    logger.info("Merging for ROI %s" % block.write_roi)
    ndim = 3
    for k in range(ndim):
        myelin_affs_array[k] = myelin_array

    z_dim = 0
    if agglomerate_in_xy:
        myelin_affs_array[z_dim] = 0
    else:
        # for z direction, we'd need to the section n+1 z affinity be min with the
        # section below
        # add null section for the first section
        # TODO: add context so we don't need a null section
        null_section = np.ones_like(myelin_array[0])
        null_section = 255*null_section
        myelin_array_shifted = np.concatenate([[null_section], myelin_array[:-1]])
        myelin_affs_array[z_dim] = np.minimum(myelin_affs_array[z_dim], myelin_array_shifted)

    return myelin_affs_array


def affs_myelin_in_block(
        block,
        myelin_ds,
        agglomerate_in_xy
        ):

    # affs = affs_ds[block.write_roi].to_ndarray()
    # print(affs)
    assert myelin_ds.dtype == np.uint8
    myelin_affs_array = convert_prediction_to_affs(
        block, myelin_ds, agglomerate_in_xy)
    assert myelin_affs_array.dtype == np.uint8
    # myelin_affs_ds[block.write_roi] = myelin_affs_array
    myelin_affs = daisy.Array(
        myelin_affs_array, myelin_ds.roi, myelin_ds.voxel_size)

    return myelin_affs


def fragment_myelin_in_block(
        block,
        myelin_affs,
        # myelin_affs_ds,
        # myelin_ds,
        # merged_affs_ds,
        # downsample_xy,
        # low_threshold,
        fragments_ds,
        rag
        ):


    logger.info("Running watershed...")
    watershed_in_block(myelin_affs,
                       block,
                       rag,
                       fragments_ds,
                       fragments_in_xy=True,
                       epsilon_agglomerate=False,
                       mask=None,
                       # use_mahotas=use_mahotas,
                       )


def segment_myelin_in_block(
        block,
        fragments_ds,
        agglomerate_in_xy=False,
        merge_function="hist_quant_50"
        ):

    logger.info("Running agglomeration for %s..." % merge_function)
    total_roi = block.write_roi
    print(myelin_affs.dtype)
    assert myelin_affs.dtype == np.uint8
    rag = Graph()
    agglomerate_in_block(
        myelin_affs,
        fragments_ds,
        total_roi,
        rag,
        merge_function=merge_function
        )

    thresholds = [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    # thresholds = [.7, .8, .9]

    segmentation_dss = []

    for threshold in thresholds:

        ds_name = "volumes/3d/%s/myelin_segmentation" % merge_function + "_%.3f" % threshold
        if agglomerate_in_xy:
            ds_name = "volumes/2d/%s/myelin_segmentation" % merge_function + "_%.3f" % threshold

        segmentation_ds = daisy.prepare_ds(
            file,
            ds_name,
            fragments_ds.roi,
            fragments_ds.voxel_size,
            fragments_ds.data.dtype,
            # write_roi=block_roi_z_dim,
            compressor={'id': 'zlib', 'level': 5})

        segmentation_dss.append(segmentation_ds)

    segment(
        fragments_ds,
        roi=total_roi,
        rag=rag,
        thresholds=thresholds,
        segmentation_dss=segmentation_dss
        )


def grow_boundary(gt, steps, only_xy=False):

    print("grow_boundary for %d steps, xy=%s" % (steps, only_xy))

    if only_xy:
        assert len(gt.shape) == 3
        for z in range(gt.shape[0]):
            grow_boundary(gt[z], steps)
        return

    # get all foreground voxels by erosion of each component
    foreground = np.zeros(shape=gt.shape, dtype=np.bool)
    masked = None
    for label in np.unique(gt):
        if label == 0:
            continue
        label_mask = gt == label
        # Assume that masked out values are the same as the label we are
        # eroding in this iteration. This ensures that at the boundary to
        # a masked region the value blob is not shrinking.
        if masked is not None:
            label_mask = np.logical_or(label_mask, masked)
        eroded_label_mask = ndimage.binary_erosion(
            label_mask, iterations=steps, border_value=1)
        foreground = np.logical_or(eroded_label_mask, foreground)

    # label new background
    background = np.logical_not(foreground)
    gt[background] = 0


if __name__ == "__main__":

    file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/outputs/2019_04/cb2_synapse_cutout3_zoom4/cb2/130000/output.zarr"

    myelin_affs_ds = "volumes/myelin_affs"
    initial_3d_myelin_segmentation_ds = "volumes/conv1/3d/hist_quant_75/myelin_segmentation_0.950"
    myelin_affs_clean_ds = "volumes/myelin_affs_inverse3"
    myelin_boundary_grow_ds = "volumes/myelin_debug/myelin_boundary_grow"
    # affs_ds = "volumes/affs"
    z_steps = 1
    xy_steps = 3

    # init
    myelin_affs_ds = daisy.open_ds(file, myelin_affs_ds)
    print(myelin_affs_ds.dtype)
    assert myelin_affs_ds.dtype == np.uint8
    myelin_seg_ds = daisy.open_ds(file, initial_3d_myelin_segmentation_ds)
    total_roi = myelin_seg_ds.roi
    voxel_size = myelin_seg_ds.voxel_size
    myelin_seg = myelin_seg_ds[total_roi]
    print("Reading myelin_seg_ndarray...")
    myelin_seg_ndarray = myelin_seg.to_ndarray()

    myelin_affs_clean_ds = daisy.prepare_ds(
        file,
        myelin_affs_clean_ds,
        total_roi,
        voxel_size,
        dtype=myelin_affs_ds.dtype,
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
    # if steps_3d:
    #     grow_boundary(myelin_seg_ndarray, steps_3d, only_xy=False)
    if steps_2d:
        grow_boundary(myelin_seg_ndarray, steps_2d, only_xy=True)

    print("Writing myelin_boundary_grow_ds...")
    myelin_boundary_grow_ds[total_roi] = myelin_seg_ndarray

    # clean myelin affs using the boundary
    # myelin_non_boundary = np.logical_not(myelin_seg_ndarray)
    myelin_non_boundary = np.logical_and(myelin_seg_ndarray, 1, dtype=np.uint8)
    print("Reading myelin_seg_ndarray...")
    # myelin_affs_ndarray = myelin_affs_ds[total_roi].to_ndarray()
    # myelin_affs_ndarray[myelin_non_boundary] = 0
    # print("Writing myelin_affs_clean_ds...")
    # a = np.zeros_like(myelin_non_boundary.shape, dtype=np.uint8)
    # a[myelin_non_boundary] = 255
    # myelin_non_boundary *= 255
    myelin_non_boundary = myelin_non_boundary * 255
    myelin_affs_clean_ds[total_roi] = myelin_non_boundary

    exit(0)

