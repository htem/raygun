
import numpy as np
import logging
import malis
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("myelin_functions")


def threshold(block, myelin_prediction, val):
    # return np.place(myelin_prediction, myelin_prediction < val, [0])
    myelin_prediction[myelin_prediction < val] = 0


def convert_prediction_to_affs(
        block,
        myelin_ds,
        agglomerate_in_xy=True
        # myelin_pred_ds
        ):

    # assuming that myelin prediction is downsampled by an integer
    # myelin_affs_array = myelin_pred_ds[block.write_roi].to_ndarray()
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


def grow_boundary(gt, steps, only_xy=False):

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


def segmentation_to_affs(
        seg, affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]):

    seg = np.array(seg, dtype=np.int32)
    affinity_neighborhood = np.array(affinity_neighborhood)

    affinities = malis.seg_to_affgraph(
            seg,
            affinity_neighborhood
    ).astype(np.uint8)

    affinities = affinities * 255

    # affinities = malis.seg_to_affgraph(
    #         seg,
    #         affinity_neighborhood
    # )

    return affinities
