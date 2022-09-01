import daisy
import numpy as np
import sys
import lsd
import logging
import collections
import multiprocessing
import scipy.ndimage as ndimage
import gt_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


'''
Changelog:
5/23/19
- use "file" instead
- add option to use non skeleton corrected segmentation
- disable processing of unlabeled because it is not necessary to make labels_mask

'''


def add_merge_error_fragments(
        fragments,
        segment_array,
        slice_segments,
        rag,
        ambiguous_fragments):
    '''For each set of global segmentation, check if their local segmentation
    are the same or not. This includes unlabeled regions.'''

    '''Algorithm
        1. Find segments belonging to a segmentid
        2. For these segments, get all local id
        3. If more than one, there is at least one split
        Then
            for each segment
            get their neighbors
            if the neighbor has the same global id but different local id
            TODO....
    '''

    fragments_by_global_id = collections.defaultdict(set)
    fragment_local_id = {}
    for f in fragments:
        zyx = daisy.Coordinate(f[1])
        segid = segment_array[zyx]
        fragments_by_global_id[segid].add(f)

    for segid in fragments_by_global_id:
        # check if they all have the same id
        if segid == 0:
            continue

        fragments = fragments_by_global_id[segid]
        fragments_by_local_id = collections.defaultdict(list)
        # fragments_in_gt = set()
        for f in fragments:
            # fragments_in_gt.add(f[0])
            zyx = daisy.Coordinate(f[1])
            local_segid = slice_segments[zyx]
            fragments_by_local_id[local_segid].append(f)

        # for each set of fragments that do not have the same slice_id
        # get the fragment's neighbors
        # check if the neighbor belongs to any other local segments
        # if so, add both in the ignore list
        if len(fragments_by_local_id) > 1:
            for local_segid in fragments_by_local_id:
                local_fragments = set(
                    [f[0] for f in fragments_by_local_id[local_segid]])
                for f in local_fragments:
                    neighbors = list(rag.adj[f].keys())
                    for n in neighbors:
                        if n == f:
                            assert False
                            continue
                        if ((n not in local_fragments) and
                                (n in fragments_in_gt)):
                            if f < n:
                                ambiguous_fragments.add(tuple([f, n]))
                            else:
                                ambiguous_fragments.add(tuple([n, f]))
                            # ambiguous_fragments.add(n)


def add_split_error_fragments(
        fragments,
        segment_array,
        slice_segments,
        rag,
        ambiguous_fragments):

    '''Algorithm
        1. Find fragments belonging to a slide id
        2. For these fragments, get their global id
        3. If more than one, there is at least one split error
        Then
            for each segment
            get the neighbors that have the same slide id
            if the neighbor has a different global id, there's a split
    '''

    fragments_by_local_id = collections.defaultdict(set)
    fragment_global_id = {}
    for f in fragments:
        zyx = daisy.Coordinate(f[1])
        segid = slice_segments[zyx]
        fragments_by_local_id[segid].add(f)

    for local_segid in fragments_by_local_id:
        # check if they all have the same id
        if local_segid == 0:
            continue

        # print("Processing local segment %s" % local_segid)
        fragments = fragments_by_local_id[local_segid]
        # print("Fragments: %s" % fragments)
        fragments_by_global_id = collections.defaultdict(list)
        # all_sectionwise_fragments = set()
        for f in fragments:
            # all_sectionwise_fragments.add(f[0])
            zyx = daisy.Coordinate(f[1])
            global_segid = segment_array[zyx]
            fragments_by_global_id[global_segid].append(f)
            fragment_global_id[f[0]] = global_segid

        if len(fragments_by_global_id) > 1:
            # print("fragments_by_global_id %s" % fragments_by_global_id)
            fragments_ids = set([f[0] for f in fragments])
            for f in fragments:
                fragment_id = f[0]
                neighbors = set(rag.adj[fragment_id].keys())
                for nid in neighbors:
                    if (nid in fragments_ids
                            and fragment_global_id[fragment_id] != fragment_global_id[nid]):
                        if fragment_id < nid:
                            ambiguous_fragments.add(tuple([fragment_id, nid]))
                        else:
                            ambiguous_fragments.add(tuple([nid, fragment_id]))


            # for gt_segid in fragments_by_global_id:
            #     if gt_segid == 0:
            #         continue
            #     global_fragments_of_segid = set(
            #         [f[0] for f in fragments_by_global_id[gt_segid]])
            #     other_fragments = all_sectionwise_fragments - global_fragments_of_segid
            #     for f in global_fragments_of_segid:
            #         neighbors = set(rag.adj[f].keys())
            #         intersect = neighbors & other_fragments
            #         for n in intersect:
            #             if f < n:
            #                 ambiguous_fragments.add(tuple([f, n]))
            #             else:
            #                 ambiguous_fragments.add(tuple([n, f]))


zlib_lock = multiprocessing.Lock()
zlib_lock2 = multiprocessing.Lock()


def process_block(
        block,
        file,
        fragments_ds_path,
        segment_f,
        segment_ds_path,
        # unlabeled_f,
        # unlabeled_ds_path,
        rag_provider,
        hi_threshold_ds,
        lo_threshold_ds,
        mask_ds,
        ):

    total_roi = block.read_roi
    # print("block.read_roi: %s" % block.read_roi)

    # print("resetting mask...")
    if reset_mask:
        mask_ds[total_roi] = 0
    fragments_ds = daisy.open_ds(file, fragments_ds_path)
    total_roi = total_roi.intersect(fragments_ds.roi)
    if total_roi.empty():
        return 0

    with zlib_lock:
        gt_ndarray = None
        while gt_ndarray is None:
            segment_ds = daisy.open_ds(segment_f, segment_ds_path)
            try:
                gt_ndarray = segment_ds[block.read_roi].to_ndarray()
            except:
                print("Failed zlib read")
                gt_ndarray = None
        segment_array = daisy.Array(
            gt_ndarray, block.read_roi, segment_ds.voxel_size)

    # print("hi_threshold_ds.roi: %s" % hi_threshold_ds.roi)
    # print("fragments_ds.roi: %s" % fragments_ds.roi)
    # print("segment_ds.roi: %s" % segment_ds.roi)
    # print("total_roi.roi: %s" % total_roi)

    rag = rag_provider[total_roi]

    all_nodes = rag.node
    fragments = []
    for n in all_nodes:
        if "center_z" in all_nodes[n]:
            f = all_nodes[n]
            fragments.append(
                (n, (f["center_z"], f["center_y"], f["center_x"])))

    ambiguous_fragments = set()

    # add_merge_error_fragments(
    #     fragments,
    #     segment_array,
    #     hi_threshold_ds[total_roi],
    #     rag,
    #     ambiguous_fragments)

    add_split_error_fragments(
        fragments,
        segment_array,
        lo_threshold_ds[total_roi],
        rag,
        ambiguous_fragments)

    add_split_error_fragments(
        fragments,
        hi_threshold_ds[total_roi],
        segment_array,
        rag,
        ambiguous_fragments)

    # print("Relabeling %s" % total_roi)
    # print("Reading fragments..")
    fragments = fragments_ds[total_roi].to_ndarray()

    # print("Relabeling ambiguous regions..")
    labels_mask = np.ones_like(fragments, dtype=mask_ds.dtype)
    ambiguous_regions = get_ambiguous_boundary(ambiguous_fragments, fragments)
    labels_mask = np.logical_and(labels_mask, ambiguous_regions)

    # print("Write mask..")
    mask_ds[total_roi] = labels_mask

    # print("Masking out unlabeled regions...")
    # mask_array = mask_ds[block.read_roi].to_ndarray()
    # mask_array[gt_ndarray == 0] = 0
    # mask_ds[block.read_roi] = mask_array

    return 0


def get_ambiguous_boundary(fragment_pairs, fragments, steps=5):

    out = np.zeros_like(fragments, dtype=np.uint8)

    for m, n in fragment_pairs:

        # print("m: %d n: %d" % (m, n))

        m_label = fragments == m
        n_label = fragments == n
        m_grown = ndimage.binary_dilation(
            m_label, iterations=steps)
        m_margin = np.logical_and(m_grown, n_label)
        out = np.logical_or(out, m_margin)

        n_grown = ndimage.binary_dilation(
            n_label, iterations=steps)
        n_margin = np.logical_and(n_grown, m_label)
        out = np.logical_or(out, n_margin)

    return np.logical_not(out)


# def check_block(block, ds):

#     read_roi = ds.roi.intersect(block.read_roi)
#     if read_roi.empty():
#         return True

#     center_coord = (read_roi.get_begin() +
#                     read_roi.get_end()) / 2
#     center_values = ds[center_coord]
#     s = np.sum(center_values)

#     return s != 0


if __name__ == "__main__":

    # user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    config = gt_tools.load_config(sys.argv[1])

    debug = False
    reset_mask = True
    num_workers = 16
    # debug = True

    affs_file = config["affs_file"]
    segment_f = config["segment_file"]
    file = affs_file
    fragments_ds = daisy.open_ds(file, "volumes/fragments")
    z_hi_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_z_0.900")
    z_lo_threshold_ds = daisy.open_ds(file, "volumes/segmentation_slice_z_0.100")
    if "skeleton_file" in config:
        segment_ds_path = config["segmentation_skeleton_ds"]
    else:
        assert False
        segment_ds_path = config["segment_ds"]

    segment_ds = daisy.open_ds(segment_f, segment_ds_path)
    rag_provider = lsd.persistence.MongoDbRagProvider(
        config["db_name"],
        config["db_host"],
        mode='r+',
        edges_collection=config["db_edges_collection"])
    voxel_size = segment_ds.voxel_size
    context = 0  # nm
    # xy_step = 40
    total_roi_offset = segment_ds.roi.get_offset()
    total_roi_shape = segment_ds.roi.get_shape()
    total_roi = daisy.Roi(total_roi_offset, total_roi_shape)

    z_step = voxel_size[0]
    slice_z_shape = [x for x in total_roi_shape]
    slice_z_shape[0] = z_step
    slice_z_shape[1] = slice_z_shape[1] - 2*context
    slice_z_shape[2] = slice_z_shape[2] - 2*context
    slice_z_roi = daisy.Roi(total_roi_offset, slice_z_shape)
    slice_z_step = daisy.Coordinate((z_step, 0, 0))
    slice_z_roi_entire = [x for x in total_roi_shape]
    slice_z_roi_entire[0] = z_step
    slice_z_roi_entire = daisy.Roi((0, 0, 0), slice_z_roi_entire)

    hi_threshold_ds = z_hi_threshold_ds
    lo_threshold_ds = z_lo_threshold_ds
    slice_roi = slice_z_roi
    slice_step = slice_z_step
    slice_roi_entire = slice_z_roi_entire

    mask_ds = daisy.prepare_ds(
        file,
        "volumes/labels/labels_mask_z",
        total_roi,
        segment_ds.voxel_size,
        np.uint8,
        write_size=slice_roi_entire.get_shape(),
        compressor={'id': 'zlib', 'level': 5}
        )

    # print("resetting mask...")
    # if reset_mask:
    #     mask_ds[segment_ds.roi] = 0
    print("slice_roi_entire: %s" % slice_roi_entire)
    print("total_roi: %s" % total_roi)

    daisy.run_blockwise(
        total_roi,
        slice_roi_entire,
        slice_roi_entire,
        process_function=lambda b: process_block(
            b,
            file,
            "volumes/fragments",
            segment_f,
            segment_ds_path,
            # unlabeled_f,
            # unlabeled_ds_path,
            rag_provider,
            hi_threshold_ds,
            lo_threshold_ds,
            mask_ds,
            ),
        # check_function=lambda b: check_block(
        #     b, segmentation_dss[5]),
        # num_workers=num_workers,
        num_workers=num_workers,
        read_write_conflict=False,
        fit='valid')
