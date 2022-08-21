# import json
# import os
import logging
import numpy as np
# import sys
import daisy
# import copy
from networkx import Graph

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
    print(myelin_ds.dtype)
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

        ds_name = "volumes/clean/3d/%s/myelin_segmentation" % merge_function + "_%.3f" % threshold
        if agglomerate_in_xy:
            ds_name = "volumes/clean/2d/%s/myelin_segmentation" % merge_function + "_%.3f" % threshold

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



if __name__ == "__main__":
    file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/outputs/2019_04/cb2_synapse_cutout3_zoom4/cb2/130000/output.zarr"
    affs_ds = "volumes/myelin_affs_clean"
    # affs_ds = "volumes/clean/affs"

    agglomerate_in_xy = False

    affs_ds = daisy.open_ds(file, affs_ds)
    roi = affs_ds.roi
    block = daisy.block.Block(affs_ds.roi, affs_ds.roi, affs_ds.roi)

    myelin_affs = affs_myelin_in_block(
        block, affs_ds, agglomerate_in_xy)

    fragments_ds = "volumes/clean/3d/myelin_fragments"
    if agglomerate_in_xy:
        fragments_ds = "volumes/clean/2d/myelin_fragments"


    fragments_ds = daisy.prepare_ds(
        file,
        fragments_ds,
        roi,
        affs_ds.voxel_size,
        np.uint64,
        # write_roi=block_roi_z_dim,
        compressor={'id': 'zlib', 'level': 5})

    rag = Graph()
    fragment_myelin_in_block(
        block,
        myelin_affs,
        # myelin_affs_ds,
        # myelin_ds,
        # merged_affs_ds,
        # downsample_xy,
        # low_threshold,
        fragments_ds,
        rag
        )

    # myelin_affs_ds = daisy.prepare_ds(
    #     file,
    #     fragments_ds,
    #     roi,
    #     affs_ds.voxel_size,
    #     np.uint64,
    #     # write_roi=block_roi_z_dim,
    #     compressor={'id': 'zlib', 'level': 5})

    # merge_function = "hist_quant_10"
    # merge_function = "hist_quant_25"
    merge_function = "hist_quant_50"
    merge_function = "hist_quant_75"
    merge_function = "hist_quant_90"
    merge_function = "mean"

    for merge_function in [
            "hist_quant_10",
            "hist_quant_25",
            "hist_quant_50",
            "hist_quant_75",
            "hist_quant_90",
            "mean",
            "hist_quant_10_initmax",
            "hist_quant_25_initmax",
            "hist_quant_50_initmax",
            "hist_quant_75_initmax",
            "hist_quant_90_initmax"]:
        # segmentation_ds = daisy.prepare_ds(
        #     file,
        #     "volumes/clean/segmentation_slice_z" + "_%.3f" % threshold,
        #     fragments_ds.roi,
        #     fragments_ds.voxel_size,
        #     fragments_ds.data.dtype,
        #     write_roi=block_roi_z_dim,
        #     compressor={'id': 'zlib', 'level': 5})

        segment_myelin_in_block(
            block,
            fragments_ds,
            agglomerate_in_xy=agglomerate_in_xy,
            merge_function=merge_function
            # myelin_ds,
            # merged_affs_ds,
            # downsample_xy,
            # low_threshold,
            )

        # print("WORKER: Running with context %s" % os.environ['DAISY_CONTEXT'])
        # client_scheduler = daisy.Client()

        # while True:
        #     block = client_scheduler.acquire_block()
        #     if block is None:
        #         break

        #     merge_myelin_in_block(
        #         block,
        #         affs_ds,
        #         myelin_ds,
        #         merged_affs_ds,
        #         downsample_xy,
        #         low_threshold,
        #         )

        #     client_scheduler.release_block(block, ret=0)
