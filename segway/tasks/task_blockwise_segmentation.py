import logging
import lsd
# import numpy as np
import daisy
from daisy import Coordinate, Roi
import sys
import numpy as np

import os

# from funlib.segment.arrays import replace_values

import task_helper2 as task_helper
from task_03_agglomerate_blockwise import AgglomerateTask

# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


class BlockwiseSegmentationTask(task_helper.SlurmTask):
    '''Segment .

    These seeds are assumed to belong to the same segment.
    '''

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    out_dataset = daisy.Parameter()
    out_file = daisy.Parameter()
    block_size = daisy.Parameter()
    context = daisy.Parameter([0, 0, 0])
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    threshold = daisy.Parameter()
    edges_collection = daisy.Parameter()
    num_workers = daisy.Parameter(1)
    # global_roi_size = daisy.Parameter()
    # global_roi_offset = daisy.Parameter()
    # request_roi = daisy.Parameter()

    def process_block(self, block):

        print("Running segmentation for block %s" % block)

        # open RAG DB
        logging.info("Opening RAG DB...")
        rag_provider = lsd.persistence.MongoDbRagProvider(
            self.db_name,
            host=self.db_host,
            mode='r+',
            edges_collection=self.edges_collection)

        roi = block.write_roi
        sub_rag = rag_provider[roi]

        print(self.fragments.roi)
        print(block.write_roi)
        fragment_ndarray = self.fragments[roi].to_ndarray()
        # os._exit(1)

        segments = fragment_ndarray
        sub_rag.get_segmentation(self.threshold, segments)

        # print(segments)
        self.segment_ds[roi] = segments

        center_coord = (block.write_roi.get_begin() +
                        block.write_roi.get_end()) / 2
        center_values = self.segment_ds[center_coord]
        s = np.sum(center_values)
        print("Center value is %f" % s)

        return 0

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

        # self.global_roi = Roi(Coordinate(self.global_roi_offset),
        #                       Coordinate(self.global_roi_size))
        global_roi = self.fragments.roi

        self.segment_ds = daisy.prepare_ds(
            self.out_file,
            self.out_dataset,
            # self.global_roi,
            global_roi,
            self.fragments.voxel_size,
            self.fragments.data.dtype,
            # temporary fix until
            # https://github.com/zarr-developers/numcodecs/pull/87 gets
            # approved
            # (we want gzip to be the default)
            compressor={'id': 'zlib', 'level': 5})

        read_roi = daisy.Roi((0,)*self.fragments.roi.dims(), self.block_size)
        write_roi = daisy.Roi((0,)*self.fragments.roi.dims(), self.block_size)

        # if self.request_roi is not None:
        #     # total_roi = Coordinate(self.request_roi)
        #     total_roi = self.request_roi
        #     # global_roi = self.fragments.roi

        #     # print(total_roi)
        #     # print(global_roi)

        #     total_roi = daisy.expand_roi_to_grid(
        #         total_roi,
        #         global_roi,
        #         read_roi,
        #         write_roi)

        #     # print(total_roi)
        #     # os._exit(1)

        # else:
        #     # process all blocks
        #     assert(0)
        total_roi = self.segment_ds.roi.grow(tuple(self.context),
                                             tuple(self.context))

        self.schedule(
            total_roi,
            read_roi,
            write_roi,
            process_function=self.process_block,
            check_function=(self.check_block, lambda b: True),
            num_workers=self.num_workers,
            read_write_conflict=False,
            max_retries=0,
            fit='shrink')  # TODO: consider if other strategies valid
            # fit='valid')  # TODO: consider if other strategies valid
            # fit='overhang')  # TODO: consider if other strategies valid

    def check_block(self, block):

        logger.debug("Checking if block %s is complete..." % block.write_roi)

        if self.segment_ds.roi.intersect(block.write_roi).empty():
            logger.debug("Block outside of output ROI")
            return True

        center_coord = (block.write_roi.get_begin() +
                        block.write_roi.get_end()) / 2
        center_values = self.segment_ds[center_coord]
        s = np.sum(center_values)

        logger.debug("Sum of center values in %s is %f" % (block.write_roi, s))

        return s != 0

    def requires(self):
        return []
        # return [AgglomerateTask()]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    # 10560:14080, 263680:266752, 341504:344576
    roi = daisy.Roi(Coordinate([10560, 263680, 341504]),
                    Coordinate([3520, 3072, 3072]))
    # 341504, 263680, 10560
    # 21344, 16480, 264
    # roi = daisy.Roi(Coordinate([10560, 263680, 341504]),
    #                 Coordinate([1, 1, 1]))

    daisy.distribute(
        # [{'task': BlockwiseSegmentationTask(**user_configs, request_roi=roi),
        [{'task': BlockwiseSegmentationTask(global_config=global_config,
                                            **user_configs),
         'request': [roi]}],
        global_config=global_config)
