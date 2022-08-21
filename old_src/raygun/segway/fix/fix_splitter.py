import json
import logging
import lsd
# import os
import daisy
import sys
import datetime
import time

import networkx
# from parallel_read_rag import parallel_read_rag

import task_helper
from parallel_relabel import parallel_relabel
from task_grow_segmentation import GrowSegmentationTask

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
# logging.getLogger('task_grow_segmentation').setLevel(logging.DEBUG)


def get_connected_components(graph, threshold):
    '''Get all connected components in the RAG, as indicated by the
    'merge_score' attribute of edges.'''

    merge_graph = networkx.Graph()
    merge_graph.add_nodes_from(graph.nodes())

    for u, v, data in graph.edges(data=True):
        if (data['merge_score'] is not None and
                data['merge_score'] <= threshold):
            merge_graph.add_edge(u, v)

    components = networkx.connected_components(merge_graph)

    return [list(component) for component in components]


def fix_splitter(
        zyx0,
        zyx1,
        fragments_file,
        fragments_dataset,
        # out_file,
        # out_dataset,
        db_host,
        db_name,
        edges_collection,
        threshold,
        context,
        # roi_offset=None,
        # roi_shape=None,
        # num_workers=4,
        segment_file,
        segment_dataset,
        global_config,

        **kwargs):
    '''
    1. Adding an edge between loc0 and loc1 in the database
    2. Choose the smaller ID of the two and run seeded segmentation on the
    other loc

    # get xyz1 xyz2
    # get their fragment ids
    # get nodes from fragment id?
    # make an edge between these nodes with score = 0
    # run seeded segmentation
    '''

    # print(datetime.datetime.now())

    zyx0 = daisy.Coordinate(zyx0)
    zyx1 = daisy.Coordinate(zyx1)
    context = daisy.Coordinate(context)

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)
    # open segment_ds
    segment_dataset_f = segment_dataset + '_' + str(threshold)
    segment_ds = daisy.open_ds(segment_file, segment_dataset_f)
    # open RAG DB
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode='r',
        edges_collection=edges_collection,
        position_attribute=['center_z', 'center_y', 'center_x'])

    # get segment ids
    # segments = segment_ds[roi]
    segment0 = segment_ds[zyx0]
    segment1 = segment_ds[zyx1]
    if segment0 == segment1:
        print("Two segments are equal (%d). No need to merge." % segment0)
        exit(0)

    nodeid0 = fragments[zyx0]
    nodeid1 = fragments[zyx1]

    roi_offset = [min(k0, k1) for k0, k1 in zip(list(zyx0), list(zyx1))]
    roi_shape = [abs(k0 - k1) + 1 for k0, k1 in zip(list(zyx0), list(zyx1))]
    roi = daisy.Roi(tuple(roi_offset), tuple(roi_shape))
    # print("roi_offset: %s" % roi_offset)
    # print("roi_shape: %s" % roi_shape)
    # print("ROI: %s" % roi)
    roi = roi.grow(context, context).snap_to_grid(fragments.voxel_size)
    # print("ROI: %s" % roi)

    # print("node0 zyx: ", end='')
    # print(zyx0)
    # print("node1 zyx: ", end='')
    # print(zyx1)
    # print("ROI: %s" % roi)

    subrag = rag_provider[roi]
    # print(subrag.node)

    # make sure that both nodes reside in sub ROI
    for n in [nodeid0, nodeid1]:
        print(n)
        if n not in subrag.node:
            raise RuntimeError(
                "Sub-RAG does not contain both nodes, "
                "please increase ROI context")

    # add edge between these two nodes
    # TODO: does this overwrite correctly?
    if nodeid0 < nodeid1:
        u, v = nodeid0, nodeid1
    else:
        u, v = nodeid1, nodeid0
    subrag.add_edge(
        u, v, {'merge_score': 0.0, 'agglomerated': True, 'user': True})

    unify_segment, seed_node = ((segment0, nodeid1) if segment0 < segment1
                                else (segment1, nodeid0))

    seed_node_zyx = (subrag.node[seed_node]['center_z'],
                     subrag.node[seed_node]['center_y'],
                     subrag.node[seed_node]['center_x'])
    seed_roi = daisy.Roi(seed_node_zyx, (1, 1, 1))

    # submit growsegment task
    print("node0 segment %s zyx: " % segment0, end='')
    print(zyx0)
    print("node1 segment %s zyx: " % segment1, end='')
    print(zyx1)
    print("seed segment %s zyx: " % unify_segment, end='')
    print(seed_node_zyx)

    # exit(0)

    task = GrowSegmentationTask(
            global_config=global_config,
            segment_id=unify_segment,
            seed_zyxs=[seed_node_zyx],
            out_dataset=segment_dataset,
            threshold=threshold,
            **user_configs)

    daisy.distribute(
        [{'task': task, 'request': [seed_roi]}],
        global_config=global_config)

    exit(0)

    total_roi = fragments.roi
    if roi_offset is not None:
        assert roi_shape is not None, "If roi_offset is set, roi_shape " \
                                      "also needs to be provided"
        total_roi = daisy.Roi(offset=roi_offset, shape=roi_shape)

    print(datetime.datetime.now())
    print("Reading RAG in %s" % total_roi)
    rag = rag_provider[total_roi]
    print("Number of nodes in RAG: %d" % (len(rag.nodes())))
    print("Number of edges in RAG: %d" % (len(rag.edges())))

    # exit(0)

    print(datetime.datetime.now())
    # print("Reading fragments in %s" % total_roi)
    # fragments = fragments[total_roi]

    # segmentation_data = fragments.to_ndarray()

    out_dataset_base = out_dataset

    # create a segmentation
    for threshold in thresholds:
        print(datetime.datetime.now())
        print("Merging for threshold %f..." % threshold)
        # start = time.time()

        # rag.get_segmentation(threshold, segmentation_data)
        # components = rag.get_connected_components(threshold)
        components = get_connected_components(rag, threshold)

        print(datetime.datetime.now())
        print("Constructing dictionary from fragments to segments")
        fragments_map = {
            fragment: component[0]
            for component in components
            for fragment in component}

        # print(datetime.datetime.now())
        # store segmentation
        print(datetime.datetime.now())
        print("Writing segmentation...")

        out_dataset = out_dataset_base + "_%.3f" % threshold

        # segmentation = daisy.prepare_ds(
        #     out_file,
        #     out_dataset,
        #     fragments.roi,
        #     fragments.voxel_size,
        #     fragments.data.dtype,
        #     # temporary fix until
        #     # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
        #     # (we want gzip to be the default)
        #     compressor={'id': 'zlib', 'level':5})
        # segmentation.data[:] = segmentation_data
        parallel_relabel(
            fragments_map,
            fragments_file,
            fragments_dataset,
            total_roi,
            block_size=(4080, 4096, 4096),
            seg_file=out_file,
            seg_dataset=out_dataset,
            num_workers=num_workers,
            retry=0)

    print(datetime.datetime.now())


def to_daisy_coord(xyz):
    return [xyz[2]*40, xyz[1]*4, xyz[0]*4]


if __name__ == "__main__":
    '''Test case for fixing splitter'''

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    print(user_configs)
    print(global_config["SplitFixTask"])

    coords = [[1137, 1090, 94], [1103, 1089, 94], [1063, 1096, 94],
              [995, 1097, 94], [982, 1090, 94], [962, 1088, 94],
              [943, 1090, 94]]

    zyx0 = to_daisy_coord(coords[0])
    zyx1 = to_daisy_coord(coords[1])

    fix_splitter(
        zyx0=zyx0,
        zyx1=zyx1,
        threshold="0.200",
        global_config=global_config,
        **user_configs, **global_config["SplitFixTask"])
