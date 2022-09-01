import logging
import lsd
# import numpy as np
import collections
import daisy
from daisy import Coordinate, Roi
import sys
import numpy as np

from bson.objectid import ObjectId
from networkx import Graph, node_connected_component

import os

# from funlib.segment.arrays import replace_values

import task_helper
from task_blockwise_segmentation import BlockwiseSegmentationTask

# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def get_connected_nodes(node, nodes, edges, threshold):
    merge_graph = Graph()
    merge_graph.add_nodes_from(nodes.keys())
    for u, v, data in edges:
        if data['merge_score'] is not None and data['merge_score'] <= threshold:
            merge_graph.add_edge(u, v)
    return node_connected_component(merge_graph, node)


def get_segmentation(rag, threshold, fragments):
    # get currently connected componets
    components = rag.get_connected_components(threshold)
    print(components)
    assert(0)

    segments = list(range(1, len(components) + 1))

    # relabel fragments of the same connected components to match merged RAG
    rag.__relabel(fragments, components, segments)


class GrowSegmentationTask(task_helper.SlurmTask):
    '''Run sparse segmentation from one or more seeds positions.

    These seeds are assumed to belong to the same segment.
    '''

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    out_dataset = daisy.Parameter()
    out_file = daisy.Parameter()
    block_size = daisy.Parameter()
    context = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    grow_db_collection = daisy.Parameter()
    skipped_db_collection = daisy.Parameter()
    segment_id = daisy.Parameter()
    # num_workers = daisy.Parameter()
    # merge_function = daisy.Parameter()
    threshold = daisy.Parameter()
    seed_zyxs = daisy.Parameter()
    seed_ids = daisy.Parameter([])
    seed_db_ids = daisy.Parameter([])
    edges_collection = daisy.Parameter()

    def process_block(self, block):

        roi = block.write_roi
        read_roi = block.read_roi
        read_roi_rag = self.rag_provider[read_roi]

        logger.info("Running seeded segmentation for %s (%s)",
                    roi, read_roi)

        continuation_nodes = list()
        all_connected_nodes = set()

        # cache fragments
        fragment_ndarray = self.fragments[roi].to_ndarray()
        fragment_array = daisy.Array(
            fragment_ndarray, roi, self.fragments.voxel_size)
        # cache node db
        read_roi_rag_nodes = read_roi_rag.node
        read_roi_rag_edges = read_roi_rag.edges(data=True)
        # cache pre-gen segments
        read_roi_segments = self.segment_ds[read_roi].to_ndarray()
        read_roi_segments = daisy.Array(
            read_roi_segments, read_roi, self.fragments.voxel_size)

        skipped_seeds = list()
        for seed in self.seed_zyxs:
            # get connected components of seeds within read_roi

            seed_zyx = Coordinate(seed)
            seed_label = fragment_array[seed_zyx]
            assert(seed_label is not None)

            if not roi.contains(seed_zyx):
                # only happens when ROI is shrunken
                raise RuntimeError
                logging.warn("Seed %s not in ROI %s!" % (seed_zyx, roi))
                skipped_seeds.add((seed_label, seed_zyx))
                continue

            logging.info("Seed %s: %s" % (seed_zyx, seed_label))

            if seed_label in all_connected_nodes:
                # skip computation if already joined with a previous segment
                continue

            connected_nodes = get_connected_nodes(
                    seed_label, read_roi_rag_nodes, read_roi_rag_edges,
                    self.threshold)

            all_connected_nodes.update(set(connected_nodes))

        # get nodes in connected components outside of ROI
        # add to continuation if not segmented with our segment_id
        # optimization: only add nodes of different segments in a given
        # direction
        segments_by_direction = collections.defaultdict(set)
        for node in all_connected_nodes:

            # skip if nodes somehow have no zyx
            if len(read_roi_rag_nodes[node]) == 0:
                logger.warn("Node %s not in read_roi" % node)
                continue

            node_zyx = (read_roi_rag_nodes[node]['center_z'],
                        read_roi_rag_nodes[node]['center_y'],
                        read_roi_rag_nodes[node]['center_x'])

            if not roi.contains(node_zyx):
                logger.info("Checking segid of %s", node_zyx)
                seg_id = read_roi_segments[Coordinate(node_zyx)]
                # seg_id = self.segment_ds[Coordinate(node_zyx)]
                assert(seg_id is not None)
                if seg_id == self.segment_id:
                    continue
                # continuation_nodes.append((node, node_zyx))
                direction = Coordinate(node_zyx) / block.write_roi.get_end()
                # logger.info("Direction is %s", direction)
                if seg_id in segments_by_direction[direction]:
                    # logger.info("Segment id %s connected", seg_id)
                    continue
                else:
                    logger.info("Continuing segment id %s", seg_id)
                    segments_by_direction[direction].add(seg_id)
                    continuation_nodes.append((node, node_zyx))

        # check if initial segmentation exists
        # for zyx in self.seed_zyxs:
        #     if read_roi_segments[Coordinate(zyx)] == 0:
        #         raise RuntimeError("%s should have been segmented" % zyx)

        # # get segments
        # if not segment_exists:
        #     logger.info("Get initial segmentation")
        #     segments = fragment_ndarray
        #     sub_rag.get_segmentation(self.threshold, segments)
        #     self.segment_ds[roi] = segments
        # else:
        #     logger.info("Use old segmentation")

        segments = read_roi_segments[roi].to_ndarray()

        # join possible different labels to one
        disjointed_labels = set(
            [read_roi_segments[Coordinate(zyx)] for zyx in self.seed_zyxs])
        logger.info(disjointed_labels)

        if (len(disjointed_labels) == 1 and
                (self.segment_id in disjointed_labels)):
            logger.info("Not rewriting segmentation")

        elif 0 in disjointed_labels:
            logger.error(
                "Some nodes have no segmentation ID. "
                "Aborting and retry...")
            return 1

        else:
            # replace labels with assigned id
            # TODO: more efficient way of replacing many labels all at once?
            for l in disjointed_labels:
                np.place(segments, segments == l, self.segment_id)
            # write back to disk
            self.segment_ds[roi] = segments

        # double check that all seeds got labeled correctly
        # for zyx in self.seed_zyxs:
        #     if zyx in ignored_seeds:
        #         continue
        #     label = self.segment_ds[Coordinate(zyx)]
        #     if label != self.segment_id:
        #         logger.error(
        #             "Seed %s is incorrectly labeled %s" %
        #             (zyx, label))
        #         assert(0)

        # # check that continuation nodes are not segmented
        # for zyx in continuation_nodes:
        #     label = self.segment_ds[Coordinate(zyx)]
        #     if label == self.segment_id:
        #         logger.error(
        #             "Node %s is segmented %s" %
        #             (zyx, label))
        #         assert(0)

        self.add_grow_nodes(continuation_nodes)
        self.remove_grow_nodes(self.seed_db_ids)
        if len(skipped_seeds):
            skipped_labels, skipped_zyxs = zip(*skipped_seeds)
            self.log_to_skippped_db(skipped_labels, skipped_zyxs)

        self.rag_provider.disconnect()

        return 0

    def remove_grow_nodes(self, ids):
        if len(ids):
            db = self.rag_provider.get_db()[self.grow_db_collection]
            logging.info("Removing seeds %s" % ids)
            db.remove(
                {'_id': {'$in': [ObjectId(id) for id in ids]}})

    def add_grow_nodes(self, nodes):
        logging.info("Adding %s" % nodes)
        if len(nodes):
            db = self.rag_provider.get_db()[self.grow_db_collection]
            db.insert_many(
                [{"id": n[0], "zyx": n[1]} for n in nodes])

    def log_to_skippped_db(self, ids, zyxs):
        if len(ids):
            assert(len(ids) == len(zyxs))
            db = self.rag_provider.get_db()[self.skipped_db_collection]
            db.insert_many(
                [{"id": n[0], "zyx": n[1]} for n in zip(ids, zyxs)])

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        logging.info("Reading fragments from %s", self.fragments_file)
        self.fragments = daisy.open_ds(self.fragments_file,
                                       self.fragments_dataset,
                                       mode='r')

        # open RAG DB
        logging.info("Opening RAG DB...")
        self.rag_provider = lsd.persistence.MongoDbRagProvider(
            self.db_name,
            host=self.db_host,
            mode='r+',
            edges_collection=self.edges_collection)

        self.out_dataset = self.out_dataset + '_' + str(self.threshold)

        self.segment_ds = daisy.open_ds(
            self.out_file,
            self.out_dataset,
            mode='r+')

        # read_roi, write_roi, total_roi = self.calculate_rois(seed)
        write_roi = daisy.Roi((0, 0, 0), self.block_size)
        read_roi = write_roi.grow(tuple(self.context), tuple(self.context))
        total_roi = self.segment_ds.roi

        self.schedule(
            total_roi,
            read_roi,
            write_roi,
            process_function=self.process_block,
            check_function=(self.pre_check, lambda b: True),
            num_workers=1,
            read_write_conflict=False,
            max_retries=0,
            # fit='shrink') # TODO: consider if other strategies valid
            fit='valid')  # TODO: consider if other strategies valid

    def pre_check(self, block):
        '''If block spans outside dataset roi, skip it but also remove assigned
        seeds.'''

        if (self._daisy.total_roi.intersects(self.segment_ds.roi) !=
                self._daisy.total_roi):
            # TODO: check correctness
            # skip this block and remove seed
            self.remove_grow_nodes(self.seed_db_ids)
            self.log_to_skippped_db(self.seed_db_ids, self.seed_zyxs)

        return False

    # def calculate_rois(self, zyx):

    #     print(zyx)

    #     context = Coordinate(self.context)

    #     total_roi = daisy.expand_roi_to_grid(
    #         Roi(Coordinate(zyx), Coordinate([1, 1, 1])),
    #         read_roi,
    #         write_roi)

    #     print(total_roi)
    #     os._exit(1)

    #     return (read_roi, write_roi, total_roi)

    def requires(self):
        return [BlockwiseSegmentationTask(global_config=self.global_config)]


if __name__ == "__main__":
    # running a quick test

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    # seed = user_configs["seed_zyxs"][0]
    seed = tuple(global_config["GrowSegmentationTask"]["seed_zyxs"][0])
    # seed = [1, 1, 1]

    block_size = global_config["GrowSegmentationTask"]["block_size"]
    # context = global_config["GrowSegmentationTask"]["context"]
    write_roi = daisy.Roi((0, 0, 0), block_size)
    # read_roi = write_roi.grow(tuple(context), tuple(context))

    request_roi = daisy.expand_write_roi_to_grid(
            Roi(seed, (1, 1, 1)),
            write_roi)

    task = GrowSegmentationTask(
            global_config=global_config,
            grow_db_collection="test_grow_db",
            skipped_db_collection="test_skip_db",
            segment_id=987654,
            **user_configs)

    daisy.distribute(
        [{'task': task, 'request': [request_roi]}],
        global_config=global_config)
