import sys
import logging
import daisy
import numpy as np
import waterz
from lsd import MergeTree
import datetime
import networkx
from networkx import Graph
import copy
from funlib.segment.arrays import relabel, replace_values

from lsd.fragments import watershed_from_affinities

logger = logging.getLogger(__name__)


def watershed_in_block(
        affs,
        block,
        rag_provider,
        fragments_out,
        fragments_in_xy,
        epsilon_agglomerate,
        mask,
        use_mahotas=False):

    # total_roi = affs.roi

    logger.debug("reading affs from %s", block.read_roi)
    affs = affs.intersect(block.read_roi)
    affs.materialize()

    if mask is not None:

        logger.debug("reading mask from %s", block.read_roi)
        mask = mask.to_ndarray(affs.roi, fill_value=0)
        logger.debug("masking affinities")
        affs.data *= mask

    # extract fragments
    # print("affs.data: ", affs.data)
    fragments_data, n = watershed_from_affinities(
        affs.data,
        fragments_in_xy=fragments_in_xy,
        epsilon_agglomerate=epsilon_agglomerate,
        use_mahotas=use_mahotas)
    if mask is not None:
        fragments_data *= mask.astype(np.uint64)
    fragments = daisy.Array(fragments_data, affs.roi, affs.voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    fragments.materialize()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    max_id = fragments.data.max()
    if max_id > block.write_roi.size():
        logger.warning(
            "fragments in %s have max ID %d, relabelling...",
            block.write_roi, max_id)
        fragments.data, n = relabel(fragments.data)

    # ensure unique IDs
    size_of_voxel = daisy.Roi((0,)*affs.roi.dims(), affs.voxel_size).size()
    num_voxels_in_block = block.requested_write_roi.size()//size_of_voxel
    id_bump = block.block_id*num_voxels_in_block
    logger.debug("bumping fragment IDs by %i", id_bump)
    fragments.data[fragments.data>0] += id_bump
    # fragment_ids = range(id_bump + 1, id_bump + 1 + n)

    # store fragments
    logger.debug("writing fragments to %s", block.write_roi)
    fragments_out[block.write_roi] = fragments

def get_connected_components(rag, threshold):
    '''Get all connected components in the RAG, as indicated by the
    'merge_score' attribute of edges.'''

    merge_graph = Graph()
    merge_graph.add_nodes_from(rag.nodes())

    for u, v, data in rag.edges(data=True):
        if data['merge_score'] is not None and data['merge_score'] <= threshold:
            merge_graph.add_edge(u, v)

    components = networkx.connected_components(merge_graph)

    return [list(component) for component in components]


def get_segmentation_relabel(array, components, component_labels):

    old_values = []
    new_values = []

    for component, label in zip(components, component_labels):
        for c in component:
            old_values.append(c)
            new_values.append(label)

    array[:] = replace_values(array, old_values, new_values)


def get_segmentation(rag, threshold, fragments):
    # get currently connected componets
    components = get_connected_components(rag, threshold)
    # print(components)

    segments = list(range(1, len(components) + 1))

    # relabel fragments of the same connected components to match merged RAG
    get_segmentation_relabel(fragments, components, segments)


def agglomerate_in_block(
        affs,
        fragments,
        roi,
        rag,
        merge_function="hist_quant_50",
        threshold=1.0,
        unmerge_list=None,
        ):

    # logger.info(
    #     "Agglomerating in block %s with context of %s",
    #     block.write_roi, block.read_roi)
    merge_function = {
        'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
        'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
        'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
        'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
        'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
        'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
        'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
        'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
        'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
        'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
        'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
    }[merge_function]

    # get the sub-{affs, fragments, graph} to work on
    affs = affs.intersect(roi)
    fragments = fragments.to_ndarray(affs.roi, fill_value=0)

    # waterz uses memory proportional to the max label in fragments, therefore
    # we relabel them here and use those
    fragments_relabelled, n, fragment_relabel_map, forward_relabel_map = relabel(
        fragments,
        return_backwards_map=True,
        return_forward_map=True)

    # print(forward_relabel_map); exit()

    if unmerge_list:
        for group_tuple in unmerge_list:
            for group in group_tuple:
                for i in range(len(group)):
                    group[i] = forward_relabel_map[group[i]]

    # print(unmerge_list); exit()

    logger.debug("affs shape: %s", affs.shape)
    logger.debug("fragments shape: %s", fragments.shape)
    logger.debug("fragments num: %d", n)

    # convert affs to float32 ndarray with values between 0 and 1
    affs = affs.to_ndarray()[0:3]
    if affs.dtype == np.uint8:
        affs = affs.astype(np.float32)/255.0

    # So far, 'rag' does not contain any edges belonging to write_roi (there
    # might be a few edges from neighboring blocks, though). Run waterz until
    # threshold 0 to get the waterz RAG, which tells us which nodes are
    # neighboring. Use this to populate 'rag' with edges. Then run waterz for
    # the given threshold.

    # for efficiency, we create one waterz call with both thresholds
    generator = waterz.agglomerate(
            affs=affs,
            thresholds=[0, threshold],
            fragments=fragments_relabelled,
            scoring_function=merge_function,
            discretize_queue=256,
            return_merge_history=True,
            return_region_graph=True,
            unmerge_group_list_tuple_list=unmerge_list)

    # add edges to RAG
    _, _, initial_rag = next(generator)
    for edge in initial_rag:
        u, v = fragment_relabel_map[edge['u']], fragment_relabel_map[edge['v']]
        # this might overwrite already existing edges from neighboring blocks,
        # but that's fine, we only write attributes for edges within write_roi
        rag.add_edge(u, v, merge_score=None, agglomerated=True)

    # agglomerate fragments using affs
    _, merge_history, _ = next(generator)

    # cleanup generator
    for _, _, _ in generator:
        pass

    # create a merge tree from the merge history
    merge_tree = MergeTree(fragment_relabel_map)
    for merge in merge_history:

        a, b, c, score = merge['a'], merge['b'], merge['c'], merge['score']
        merge_tree.merge(
            fragment_relabel_map[a],
            fragment_relabel_map[b],
            fragment_relabel_map[c],
            score)

    # mark edges in original RAG with score at time of merging
    logger.debug("marking merged edges...")
    num_merged = 0
    for u, v, data in rag.edges(data=True):
        merge_score = merge_tree.find_merge(u, v)
        data['merge_score'] = merge_score
        if merge_score is not None:
            num_merged += 1

    logger.info("merged %d edges", num_merged)


def segment(
        fragments,
        roi,
        rag,
        thresholds,
        segmentation_dss
        ):

    logging.info("Reading fragments and RAG in %s" % roi)
    fragments = fragments[roi]
    logging.info("Number of nodes in RAG: %d" % (len(rag.nodes())))
    logging.info("Number of edges in RAG: %d" % (len(rag.edges())))

    logging.info(
        "%s: loaded to memory" % datetime.datetime.now()
        )
    # create a segmentation
    logging.info("Merging...")
    segments = fragments.to_ndarray()

    for threshold, segmentation in zip(thresholds, segmentation_dss):

        segmentation_data = copy.deepcopy(segments)
        get_segmentation(rag, threshold, segmentation_data)
        logging.info(
            "%s: merged" % datetime.datetime.now()
            )
        logging.info("Writing segmentation for threshold %f..." % threshold)
        segmentation[roi] = segmentation_data


def pooling(mat, ksize, method, pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.

    Adapted from https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    '''

    # m, n = mat.shape[:2]
    # assume that yx is the last two indices in the array
    shape_len = len(mat.shape)
    # assert len(mat.shape) == 4
    m, n = mat.shape[shape_len-2], mat.shape[shape_len-1]
    ky, kx = ksize

    # ceil = lambda x, y: int(np.ceil(x/float(y)))

    # if pad:
    #     assert False, "Untested"
    #     ny = ceil(m, ky)
    #     nx = ceil(n, kx)
    #     size = (ny*ky, nx*kx)+mat.shape[2:]
    #     mat_pad = np.full(size,np.nan)
    #     mat_pad[:m, :n, ...] = mat
    # else:
    #     ny = m//ky
    #     nx = n//kx
    #     mat_pad = mat[..., :ny*ky, :nx*kx]

    ny = m//ky
    nx = n//kx
    # new_shape = (mat.shape[0], mat.shape[1]) + (ny, ky, nx, kx)
    new_shape = tuple([mat.shape[k] for k in range(shape_len-2)]) + (ny, ky, nx, kx)

    if method == 'max':
        result = np.nanmax(mat.reshape(new_shape), axis=(shape_len-1, shape_len+1))
    elif method == 'min':
        result = np.nanmin(mat.reshape(new_shape), axis=(shape_len-1, shape_len+1))
    elif method == 'mean':
        result = np.nanmean(mat.reshape(new_shape), axis=(shape_len-1, shape_len+1))
    else:
        raise RuntimeError("Unsupported method %s" % method)

    return result


def downsample_with_pooling(fullres_ds, downsample_xy, mode='min'):

    ds_array = fullres_ds
    if fullres_ds is not None:
        if downsample_xy > 1:
            voxel_ds_factors = (1, downsample_xy, downsample_xy)
            fullres_ndarray = fullres_ds.to_ndarray()
            ds_ndarray = pooling(fullres_ndarray, (downsample_xy, downsample_xy), mode)
            ds_array = daisy.Array(
                ds_ndarray,
                roi=fullres_ds.roi,
                voxel_size=fullres_ds.voxel_size*voxel_ds_factors
                )

    return ds_array
