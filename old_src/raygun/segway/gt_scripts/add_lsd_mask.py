import logging
# from gunpowder import BatchFilter, Array
# from scipy.ndimage import gaussian_filter
# from scipy.ndimage.filters import convolve
# from numpy.lib.stride_tricks import as_strided
import numpy as np
# import time
import sys

import daisy

import gt_tools

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def conv_ndarray(ndarray, z_step=True, xy_step=False):

    orig = ndarray.copy()
    if z_step:
        for i in range(len(ndarray)):
            if i != 0:
                ndarray[i] = np.minimum(orig[i-1], ndarray[i])
            if i != len(ndarray)-1:
                ndarray[i] = np.minimum(orig[i+1], ndarray[i])

    if xy_step:
        y_range = len(ndarray[0])
        for i in range(y_range):
            if i != 0:
                ndarray[:, i, :] = np.minimum(orig[:, i-1, :], ndarray[:, i, :])
            if i != y_range-1:
                ndarray[:, i, :] = np.minimum(orig[:, i+1, :], ndarray[:, i, :])

        x_range = len(ndarray[0][0])
        for i in range(x_range):
            if i != 0:
                ndarray[:, :, i] = np.minimum(orig[:, :, i-1], ndarray[:, :, i])
            if i != x_range-1:
                ndarray[:, :, i] = np.minimum(orig[:, :, i+1], ndarray[:, :, i])

    return ndarray

def add_lsd_mask_blockwise(
        block,
        xy_step,
        z_step,
        mask_ds,
        unlabeled_ds,
        lsd_mask_ds,
        lsd_unlabeled_ds,
        ):

    # segmentation_array = batch.arrays[self.segmentation]
    # segmentation_array = mask_ds[block.read_roi]

    intersect_roi = block.read_roi.intersect(mask_ds.roi)

    # for in_ds, out_ds in zip([unlabeled_ds], [lsd_unlabeled_ds]):

    ndarray = np.zeros(block.read_roi.get_shape(), dtype=np.uint8)
    array = daisy.Array(ndarray, block.read_roi, mask_ds.voxel_size)
    array[intersect_roi] = mask_ds[intersect_roi]
    ndarray = array.to_ndarray()
    xyz_step = min(xy_step, z_step)
    for i in range(xyz_step):
        conv_ndarray(ndarray, z_step=True, xy_step=True)
    for i in range(xy_step-xyz_step):
        conv_ndarray(ndarray, z_step=False, xy_step=True)
    for i in range(z_step-xyz_step):
        conv_ndarray(ndarray, z_step=True, xy_step=False)
    lsd_mask_ds[block.write_roi] = array[block.write_roi]

    # ndarray = np.zeros(block.read_roi.get_shape(), dtype=np.uint8)
    # array = daisy.Array(ndarray, block.read_roi, unlabeled_ds.voxel_size)
    array[array.roi] = 0
    array[intersect_roi] = unlabeled_ds[intersect_roi]
    for i in range(z_step):
        conv_ndarray(ndarray, z_step=True, xy_step=False)
    lsd_unlabeled_ds[block.write_roi] = array[block.write_roi]


if __name__ == "__main__":

    config = gt_tools.load_config(sys.argv[1])
    file = config["out_file"]

    # segment_ds = daisy.open_ds(
    #     file,
    #     "volumes/labels/neuron_ids")

    labels_mask_ds = daisy.open_ds(
        file,
        "volumes/labels/labels_mask2")
    unlabeled_ds = daisy.open_ds(
        file,
        "volumes/labels/unlabeled")

    # add_lsd = AddLocalShapeDescriptor(
    #     sigma=80,
    #     voxel_size=segment_ds.voxel_size,
    #     )

    # write_roi = daisy.Roi((0, 0, 0), (240, 256, 256))
    write_roi = daisy.Roi((0, 0, 0), (480, 512, 512))
    context = daisy.Coordinate((80*2, 80*2, 80*2))
    read_roi = write_roi.grow(context, context)
    read_roi = read_roi.snap_to_grid(
        labels_mask_ds.voxel_size, mode='grow')
    total_roi = labels_mask_ds.roi
    total_roi = total_roi.grow(context, context)

    lsd_mask_ds = daisy.prepare_ds(
        file,
        "volumes/labels/lsd_mask",
        labels_mask_ds.roi,
        labels_mask_ds.voxel_size,
        np.uint8,
        write_size=write_roi.get_shape(),
        compressor={'id': 'zlib', 'level': 5}
        )

    lsd_unlabeled_ds = daisy.prepare_ds(
        file,
        "volumes/labels/lsd_unlabeled_mask",
        labels_mask_ds.roi,
        labels_mask_ds.voxel_size,
        np.uint8,
        write_size=write_roi.get_shape(),
        compressor={'id': 'zlib', 'level': 5}
        )

    print("Total ROI: ", total_roi)
    print("LSD write_roi: ", write_roi)
    print("LSD context: ", context)
    print("LSD read_roi: ", read_roi)
    # exit(0)
    xy_step = int(80*2/4)
    z_step = int(80*2/40)

    # process block-wise
    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: add_lsd_mask_blockwise(
            b,
            xy_step,
            z_step,
            labels_mask_ds,
            unlabeled_ds,
            lsd_mask_ds,
            lsd_unlabeled_ds,
            ),
        num_workers=24,
        # num_workers=1,
        read_write_conflict=False,
        fit='shrink')

    # add_lsd.add_lsd_blockwise(
    #     block, segment_ds, lsd_mask_ds)

