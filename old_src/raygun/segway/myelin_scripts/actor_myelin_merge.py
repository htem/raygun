import json
import os
import logging
import numpy as np
import sys
import daisy
from .. import myelin_postprocess
from .myelin_functions import convert_prediction_to_affs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("actor_myelin_merge")


def merge_myelin_in_block(
        block,
        file,
        affs_ds,
        myelin_ds,
        merged_affs_ds,
        downsample_xy,
        run_postprocess=False,
        ):

    logger.info("Begin merge_myelin_in_block")
    # # assuming that myelin prediction is downsampled by an integer
    affs_array = affs_ds[block.read_roi].to_ndarray()
    # myelin_array = myelin_ds[block.read_roi].to_ndarray()

    # # thresholding
    # np.place(myelin_array, myelin_array < low_threshold, [0])

    # myelin_postprocess(block, file, myelin_ds, myelin_affs)

    logger.info("Running post processing...")
    if run_postprocess:
        myelin_affs = myelin_postprocess(block, file, myelin_ds)
    else:
        # myelin_affs = daisy.Array(
        #     np.ndarray(myelin_affs_shape, dtype=np.uint8), block.read_roi, myelin_ds.voxel_size)
        # myelin_affs = daisy.open_ds(file, myelin_ds)
        myelin_affs_ndarray = convert_prediction_to_affs(
            myelin_ds[block.read_roi].to_ndarray(),
            agglomerate_in_xy=False,  # necessary for 3D agglomeration
            )
        # myelin_affs_shape = tuple([3] + [k for k in myelin_ds.shape])
        myelin_affs = daisy.Array(
            myelin_affs_ndarray, block.read_roi, myelin_ds.voxel_size)

    # print("myelin_affs.to_ndarray(): %s" % myelin_affs.to_ndarray())
    # exit(0)
    logger.info("Up-sampling myelin array...")
    myelin_affs_array = myelin_affs.to_ndarray()
    y_axis = 2
    x_axis = 3
    myelin_affs_array = np.repeat(myelin_affs_array, downsample_xy, axis=y_axis)
    myelin_affs_array = np.repeat(myelin_affs_array, downsample_xy, axis=x_axis)
    for n, m in zip(affs_array.shape[2:3], myelin_affs_array.shape[2:3]):
        assert m == n, "affs_array.shape %s is not same as myelin_affs_array.shape %s" % (affs_array.shape, myelin_affs_array.shape)

    logger.info("Merging for ROI %s" % block.read_roi)
    # ndim = 3
    # for k in range(ndim):
    #     affs_array[k] = np.minimum(affs_array[k], myelin_affs_array)
    # print("affs_array: %s" % affs_array)
    # print("myelin_affs_array: %s" % myelin_affs_array)
    affs_array = np.minimum(affs_array, myelin_affs_array)

    # for z direction, we'd need to the section n+1 z affinity be min with the
    # section below
    # add null section for the first section
    # TODO: add context so we don't need a null section
    # null_section = np.ones_like(myelin_affs_array[0])
    # null_section = 255*null_section
    # myelin_array_shifted = np.concatenate([[null_section], myelin_affs_array[:-1]])
    # z_dim = 0
    # affs_array[z_dim] = np.minimum(affs_array[z_dim], myelin_array_shifted)

    logger.info("Writing merged results...")
    merged_affs_ds[block.write_roi] = affs_array


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    affs_file = None
    affs_dataset = None
    myelin_file = None
    myelin_dataset = None
    merged_affs_file = None
    merged_affs_dataset = None
    downsample_xy = None

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    affs_ds = daisy.open_ds(affs_file, affs_dataset)
    myelin_ds = daisy.open_ds(myelin_file, myelin_dataset)
    merged_affs_ds = daisy.open_ds(
        merged_affs_file, merged_affs_dataset, mode="r+")
    assert merged_affs_ds.data.dtype == np.uint8
    assert(downsample_xy is not None)

    print("WORKER: Running with context %s" % os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        merge_myelin_in_block(
            block,
            affs_file,
            affs_ds,
            myelin_ds,
            merged_affs_ds,
            downsample_xy,
            )

        client_scheduler.release_block(block, ret=0)
