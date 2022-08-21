
import numpy as np
import logging
from scipy import ndimage
import daisy
import networkx

from .segmentation_functions import agglomerate_in_block, segment, \
    watershed_in_block, grow_boundary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("myelin_functions")


def thresholding(arrays, block, in_key, out_key, low_threshold):

    myelin_prediction = arrays[in_key]
    print(block)
    myelin_prediction = myelin_prediction[block.read_roi].to_ndarray()
    # return np.place(myelin_prediction, myelin_prediction < low_threshold, [0])
    myelin_prediction[myelin_prediction < low_threshold] = 0
    ret = myelin_prediction.copy()

    ret_array = daisy.Array(
        ret, block.read_roi, arrays[in_key].voxel_size)

    arrays[out_key] = ret_array


def convert_prediction_to_affs(
        # block,
        # myelin_ds,
        prediction_array,
        agglomerate_in_xy=True
        # myelin_pred_ds
        ):

    # assuming that myelin prediction is downsampled by an integer
    # myelin_affs_array = myelin_pred_ds[block.write_roi].to_ndarray()
    # prediction_array = myelin_ds[block.write_roi].to_ndarray()
    affs_shape = tuple([3] + list(prediction_array.shape))
    myelin_affs_array = np.zeros(affs_shape, dtype=prediction_array.dtype)

    # # thresholding
    # low_threshold = 128
    # np.place(prediction_array, prediction_array < low_threshold, [0])

    # logger.info("Merging for ROI %s" % block.write_roi)
    ndim = 3
    for k in range(ndim):
        myelin_affs_array[k] = prediction_array

    z_dim = 0
    if agglomerate_in_xy:
        myelin_affs_array[z_dim] = 0
    else:
        # for z direction, we'd need to the section n+1 z affinity be min with the
        # section below
        # add null section for the first section
        # TODO: add context so we don't need a null section
        null_section = np.ones_like(prediction_array[0])
        null_section = 255*null_section
        myelin_array_shifted = np.concatenate([[null_section], prediction_array[:-1]])
        myelin_affs_array[z_dim] = np.minimum(myelin_affs_array[z_dim], myelin_array_shifted)

    return myelin_affs_array


def affs_myelin_in_block(
        arrays,
        block,
        in_key, out_key,
        # myelin_ds,
        agglomerate_in_xy
        ):

    # affs = affs_ds[block.write_roi].to_ndarray()
    # print(affs)
    # print(myelin_ds.dtype)
    pred_array = arrays[in_key].to_ndarray()
    assert pred_array.dtype == np.uint8
    myelin_affs_array = convert_prediction_to_affs(
        pred_array, agglomerate_in_xy)
    assert myelin_affs_array.dtype == np.uint8
    # myelin_affs_ds[block.write_roi] = myelin_affs_array
    # myelin_affs = daisy.Array(
    #     myelin_affs_array, myelin_ds.roi, myelin_ds.voxel_size)

    affs_ds = daisy.Array(
        myelin_affs_array,
        block.read_roi, arrays[in_key].voxel_size)

    arrays[out_key] = affs_ds


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

    import malis
    affinities = malis.seg_to_affgraph(
            seg,
            affinity_neighborhood
    ).astype(np.uint8)

    affinities = affinities * 255

    return affinities


def make_affs_from_segments(
        arrays,
        block,
        in_key,
        out_key,
        ):

    seg = arrays[in_key].to_ndarray()
    affs = segmentation_to_affs(seg)

    out = daisy.Array(
        affs, block.read_roi, arrays[in_key].voxel_size)

    arrays[out_key] = out


def fragment_myelin_in_block(
        arrays,
        block,
        # voxel_size,
        in_key,
        out_fragment_key,
        # myelin_affs,
        # myelin_affs_ds,
        # myelin_ds,
        # merged_affs_ds,
        # downsample_xy,
        # low_threshold,
        # fragments_ds,
        ):

    roi = block.read_roi
    # affs = daisy.Array(arrays[in_key], roi, voxel_size)
    affs = arrays[in_key]
    fragments_array = daisy.Array(
        np.ndarray(affs.shape[1:], dtype=np.uint64), roi, affs.voxel_size)

    print(fragments_array.shape)
    print(affs.shape)
    # exit(0)

    print("Running watershed...")
    watershed_in_block(affs,
                       block,
                       # rag,
                       fragments_array,
                       fragments_in_xy=True,
                       epsilon_agglomerate=False,
                       mask=None,
                       # use_mahotas=use_mahotas,
                       )

    assert len(fragments_array.shape) == 3
    arrays[out_fragment_key] = fragments_array


def segment_myelin(
        arrays,
        block,
        fragment_key,
        rag_key,
        segment_key,
        threshold,
        ):

    segment_array = daisy.Array(
        np.ndarray(arrays[fragment_key].shape, dtype=np.uint64),
        block.read_roi, arrays[fragment_key].voxel_size)

    segment(
        arrays[fragment_key],
        roi=block.read_roi,
        rag=arrays[rag_key],
        thresholds=[threshold],
        segmentation_dss=[segment_array]
        )

    arrays[segment_key] = segment_array
    # print(segment_array.shape)
    # print(arrays[fragment_key].shape)


def agglomerate_myelin(
        arrays,
        block,
        affs_key,
        fragment_key,
        out_rag_key,
        merge_function,
        ):

    rag = networkx.Graph()
    agglomerate_in_block(
        arrays[affs_key],
        arrays[fragment_key],
        block.read_roi,
        rag,
        merge_function=merge_function
        )

    arrays[out_rag_key] = rag


def grow_segment_boundary(
        arrays,
        block,
        segment_key,
        z_steps,
        xy_steps,
        ):

    segment_ds = arrays[segment_key]
    segment_ndarray = segment_ds.to_ndarray()
    # perform boundary grow of the segment
    # steps to perform boundary grow in pixels
    steps_3d = min(z_steps, xy_steps)
    steps_2d = max(z_steps, xy_steps) - steps_3d
    if steps_3d:
        grow_boundary(segment_ndarray, steps_3d, only_xy=False)
    if steps_2d:
        grow_boundary(segment_ndarray, steps_2d, only_xy=True)

    segment_ds[segment_ds.roi] = segment_ndarray


def clean_pred_with_segment_boundary(
        arrays,
        block,
        segment_key,
        pred_key,
        pred_out_key,
        ):

    # print("Writing myelin_boundary_grow_ds...")
    # myelin_boundary_grow_ds[block.read_roi] = myelin_seg_ndarray
    # clean myelin affs using the boundary
    # myelin_non_boundary = np.logical_not(myelin_seg_ndarray)
    segment = arrays[segment_key].to_ndarray()
    non_boundary = np.logical_and(segment, 1, dtype=np.uint8)
    # myelin_non_boundary = myelin_non_boundary * 255
    # myelin_affs_clean_ds[block.read_roi] = myelin_non_boundary
    pred_ds = arrays[pred_key]
    # pred_ndarray = pred_ds[pred_ds.roi].to_ndarray()
    pred_ndarray = pred_ds.to_ndarray()
    pred_ndarray[non_boundary] = 255

    clean_pred_array = daisy.Array(
        pred_ndarray, block.read_roi, pred_ds.voxel_size)

    arrays[pred_out_key] = clean_pred_array
    # pred_ds[pred_ds.roi] = pred_ndarray


def extract_boundary(
        arrays,
        block,
        in_key,
        out_key,
        ):

    # print("Writing myelin_boundary_grow_ds...")
    # myelin_boundary_grow_ds[block.read_roi] = myelin_seg_ndarray
    # clean myelin affs using the boundary
    # myelin_non_boundary = np.logical_not(myelin_seg_ndarray)
    segment = arrays[in_key].to_ndarray()
    non_boundary = np.logical_and(segment, 1, dtype=np.uint8)
    non_boundary = non_boundary * 255
    # myelin_affs_clean_ds[block.read_roi] = myelin_non_boundary
    pred_ds = arrays[out_key]
    # pred_ndarray = pred_ds[pred_ds.roi].to_ndarray()
    # pred_ndarray[non_boundary] = 255
    pred_ds[pred_ds.roi] = non_boundary


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


def conv(
        arrays,
        block,
        in_key,
        out_key,
        z_steps,
        xy_steps,
        ):

    ndarray = arrays[in_key].to_ndarray()
    for i in range(z_steps):
        ndarray = conv_ndarray(
            ndarray, z_step=True, xy_step=False)
    for i in range(xy_steps):
        ndarray = conv_ndarray(
            ndarray, z_step=False, xy_step=True)

    arrays[out_key][arrays[out_key].roi] = ndarray
