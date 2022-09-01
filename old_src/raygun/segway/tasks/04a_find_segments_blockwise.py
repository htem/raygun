import json
import logging
import sys

import daisy
import time
import os

import pymongo
import numpy as np
from funlib.segment.graphs.impl import connected_components

import task_helper

# logging.getLogger('daisy.persistence.shared_graph_provider').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def read_block(graph_provider, block):

    logger.debug("Reading graph in block %s", block)
    start = time.time()
    graph = graph_provider[block.read_roi]
    logger.debug("Read graph from graph provider in %.3fs",
                time.time() - start)

    nodes = {
        'id': []
    }
    edges = {
        'u': [],
        'v': []
    }

    start = time.time()
    for node, data in graph.nodes(data=True):

        # skip over nodes that are not part of this block (they have been
        # pulled in by edges leaving this block and don't have a position
        # attribute)

        if type(graph_provider.position_attribute) == list:
            probe = graph_provider.position_attribute[0]
        else:
            probe = graph_provider.position_attribute
        if probe not in data:
            continue

        nodes['id'].append(np.uint64(node))
        for k, v in data.items():
            if k not in nodes:
                nodes[k] = []
            nodes[k].append(v)

    for u, v, data in graph.edges(data=True):

        edges['u'].append(np.uint64(u))
        edges['v'].append(np.uint64(v))
        for k, v in data.items():
            if k not in edges:
                edges[k] = []
            edges[k].append(v)

    if len(nodes['id']) == 0:
        logger.debug("Graph is empty")
        return

    if len(edges['u']) == 0:
        # no edges in graph, make sure empty np array has correct dtype
        edges['u'] = np.array(edges['u'], dtype=np.uint64)
        edges['v'] = np.array(edges['v'], dtype=np.uint64)

    nodes = {
        k: np.array(v)
        for k, v in nodes.items()
    }
    edges = {
        k: np.array(v)
        for k, v in edges.items()
    }
    logger.debug("Parsed graph in %.3fs", time.time() - start)

    # start = time.time()
    return (nodes, edges)


def find_segments(
        db_host,
        db_name,
        fragments_file,
        lut_dir,
        edges_collection,
        merge_function,
        block,
        # roi_offset,
        # roi_shape,
        thresholds,
        run_type=None,
        block_id=None,
        ignore_degenerates=False,
        **kwargs):

    '''

    Args:

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        fragments_file (``string``):

            Path to the file containing the fragments.

        edges_collection (``string``):

            The name of the MongoDB database collection to use.

        roi_offset (array-like of ``int``):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

    '''

    if db_file_name is not None:
        graph_provider = daisy.persistence.FileGraphProvider(
            directory=os.path.join(db_file, db_file_name),
            chunk_size=None,
            mode='r+',
            directed=False,
            position_attribute=['center_z', 'center_y', 'center_x'],
            save_attributes_as_single_file=True,
            roi_offset=filedb_roi_offset,
            edges_roi_offset=filedb_edges_roi_offset,
            nodes_chunk_size=filedb_nodes_chunk_size,
            edges_chunk_size=filedb_edges_chunk_size,
            )
    else:
        graph_provider = daisy.persistence.MongoDbGraphProvider(
            db_name,
            db_host,
            edges_collection=edges_collection,
            position_attribute=[
                'center_z',
                'center_y',
                'center_x'],
            indexing_block_size=indexing_block_size,
            )

    res = read_block(graph_provider, block)

    if res is None:
        if not ignore_degenerates:
            raise RuntimeError('No nodes found in %s' % block)
        else:
            logger.info('No nodes found in %s' % block)
        # create dummy nodes
        node_attrs = {
            'id': np.array([0], dtype=np.uint64),
        }
        edge_attrs = {
            'u': np.array([0]),
            'v': np.array([0]),
        }
    else:
        node_attrs, edge_attrs = res

    if 'id' not in node_attrs:
        if not ignore_degenerates:
            raise RuntimeError('No nodes found in %s' % block)
        else:
            logger.info('No nodes found in %s' % block)
        nodes = [0]
    else:
        nodes = node_attrs['id']

    u_array = edge_attrs['u'].astype(np.uint64)
    v_array = edge_attrs['v'].astype(np.uint64)
    edges = np.stack([u_array, v_array], axis=1)

    logger.info("RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))

    if len(u_array) == 0:
        # this block somehow has no edges, or is not agglomerated
        u_array = np.array([0], dtype=np.uint64)
        v_array = np.array([0], dtype=np.uint64)
        edges = np.array([[0, 0]], dtype=np.uint64)

        print(f"ERROR: block {block_id} somehow has no edges, or is not agglomerated")

        return 1

    if 'merge_score' in edge_attrs:
        scores = edge_attrs['merge_score'].astype(np.float32)
    else:
        scores = np.ones_like(u_array, dtype=np.float32)

    # each block should have at least one node, edge, and score
    assert len(nodes)
    assert len(edges)
    assert len(scores)

    out_dir = os.path.join(
        fragments_file,
        lut_dir)

    if run_type:
        out_dir = os.path.join(out_dir, run_type)

    for threshold in thresholds:
        get_connected_components(
                nodes,
                edges,
                scores,
                threshold,
                merge_function,
                out_dir,
                block_id,
                ignore_degenerates=ignore_degenerates)


def get_connected_components(
        nodes,
        edges,
        scores,
        threshold,
        merge_function,
        out_dir,
        block_id=None,
        hi_threshold=0.95,
        ignore_degenerates=False,
        **kwargs):

    if block_id is None:
        block_id = 0

    logger.debug("Getting CCs for threshold %.3f..." % threshold)

    edges_tmp = edges[scores <= threshold]
    scores_tmp = scores[scores <= threshold]

    if len(edges_tmp):
        components = connected_components(nodes, edges_tmp, scores_tmp, threshold,
                                          use_node_id_as_component_id=1)

    else:
        if len(scores) == 0:
            print("edges_tmp: ", edges_tmp)
            print("scores_tmp: ", scores_tmp)
            print("edges: ", edges)
            print("scores: ", scores)
            print("len(nodes): ", len(nodes))
            if not ignore_degenerates:
                raise RuntimeError(
                    'Empty edges in graph! Likely unfinished agglomeration.')
            else:
                logger.info(
                    'Empty edges in graph! Likely unfinished agglomeration.')
        components = nodes

    lut = np.array([nodes, components])

    lookup = 'seg_frags2local_%s_%d/%d' % (merge_function, int(threshold*100), block_id)
    out_file = os.path.join(out_dir, lookup)
    np.savez_compressed(out_file, fragment_segment_lut=lut)

    unique_components = np.unique(components)
    lookup = 'nodes_%s_%d/%d' % (merge_function, int(threshold*100), block_id)
    out_file = os.path.join(out_dir, lookup)
    np.savez_compressed(out_file, nodes=unique_components)

    nodes_in_vol = set(nodes)

    def not_in_graph(u, v):
        return u not in nodes_in_vol or v not in nodes_in_vol

    logger.debug("Num edges original: ", len(edges))
    outward_edges = np.array([not_in_graph(n[0], n[1]) for n in edges])
    edges = edges[np.logical_and(scores <= threshold, outward_edges)]

    # replace IDs in edges with agglomerated IDs
    frags2seg = {n: k for n, k in np.dstack((lut[0], lut[1]))[0]}
    for i in range(len(edges)):
        if edges[i][0] in frags2seg:
            if edges[i][0] != frags2seg[edges[i][0]]:
                edges[i][0] = frags2seg[edges[i][0]]
        if edges[i][1] in frags2seg:
            if edges[i][1] != frags2seg[edges[i][1]]:
                edges[i][1] = frags2seg[edges[i][1]]

    if len(edges):
        # np.unique doesn't work on empty arrays
        edges = np.unique(edges, axis=0)

    logger.debug("Num edges pruned: ", len(edges))

    lookup = 'edges_local2frags_%s_%d/%d' % (merge_function, int(threshold*100), block_id)
    out_file = os.path.join(out_dir, lookup)
    np.savez_compressed(out_file, edges=edges)


if __name__ == "__main__":

    if sys.argv[1] == 'run':

        user_configs, global_config = task_helper.parseConfigs(sys.argv[2:])
        config = user_configs
        config.update(global_config["FindSegmentsBlockwiseTask"])

        find_segments(**config)

    else:

        print(sys.argv)
        config_file = sys.argv[1]
        with open(config_file, 'r') as f:
            run_config = json.load(f)

        for key in run_config:
            globals()['%s' % key] = run_config[key]

        if run_config.get('block_id_add_one_fix', False):
            daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

        print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
        client_scheduler = daisy.Client()

        db_client = pymongo.MongoClient(db_host)
        db = db_client[db_name]
        completion_db = db[completion_db_name]

        while True:
            block = client_scheduler.acquire_block()
            if block is None:
                break

            logger.info("Block: %s" % block)

            find_segments(
                db_host=db_host,
                db_name=db_name,
                fragments_file=fragments_file,
                lut_dir=lut_dir,
                edges_collection=edges_collection,
                merge_function=merge_function,
                block=block,
                thresholds=thresholds,
                block_id=block.block_id,
                ignore_degenerates=ignore_degenerates,
                )

            # recording block done in the database
            document = dict()
            document.update({
                'block_id': block.block_id,
                'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
                'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
                'start': 0,
                'duration': 0
            })
            completion_db.insert(document)

            client_scheduler.release_block(block, ret=0)
