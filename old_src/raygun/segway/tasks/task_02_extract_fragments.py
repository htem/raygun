# import json
import logging
import numpy as np
import os
import sys

import daisy

import task_helper2 as task_helper
from task_01_predict_blockwise import PredictTask


# logging.getLogger('daisy.persistence.file_graph_provider').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


class ExtractFragmentTask(task_helper.SlurmTask):

    '''
    Parameters:

        affs_file,
        affs_dataset,
        mask_file,
        mask_dataset (``string``):

            Where to find the affinities and mask (optional).

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction and agglomeration,
            in world units.

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        num_workers (``int``):

            How many blocks to run in parallel.

        fragments_in_xy (``bool``):

            Extract fragments section-wise.

        mask_fragments (``bool``):

            Whether to mask fragments for a specified region. Requires that the
            original sample dataset contains a dataset ``volumes/labels/mask``.
    '''

    affs_file = daisy.Parameter()
    affs_dataset = daisy.Parameter()
    block_size = daisy.Parameter()
    indexing_block_size = daisy.Parameter(None)
    context = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    db_file = daisy.Parameter(None)
    db_file_name = daisy.Parameter("db_file")
    filedb_roi_offset = daisy.Parameter(None)
    num_workers = daisy.Parameter()

    # sub_roi is used to specify the region of interest while still allocating
    # the entire input raw volume. It is useful when there is a chance that
    # sub_roi will be increased in the future.
    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)

    mask_fragments = daisy.Parameter(default=False)
    mask_file = daisy.Parameter(default=None)
    mask_dataset = daisy.Parameter(default=None)

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    fragments_in_xy = daisy.Parameter()

    raw_file = daisy.Parameter(None)
    raw_dataset = daisy.Parameter(None)

    epsilon_agglomerate = daisy.Parameter(default=0)
    use_mahotas = daisy.Parameter()

    use_myelin_net = daisy.Parameter(default=False)

    overwrite_mask_f = daisy.Parameter(None)
    overwrite_sections = daisy.Parameter(None)

    min_seed_distance = daisy.Parameter(None)  # default seed size from Jan

    force_exact_write_size = daisy.Parameter(False)

    capillary_pred_file = daisy.Parameter(None)
    capillary_pred_dataset = daisy.Parameter(None)

    filter_fragments = daisy.Parameter(0.3)

    scheduling_chunks = daisy.Parameter([1, 1, 1])
    dataset_chunks = daisy.Parameter([1, 1, 1])
    database_chunks = daisy.Parameter([1, 1, 1])

    # check block completion by counting nodes in DB
    precheck_with_db = daisy.Parameter(False)
    precheck_use_affs = daisy.Parameter(False)
    block_id_add_one_fix = daisy.Parameter(False)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        logging.info("Reading affs from %s", self.affs_file)
        self.affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        if self.mask_fragments:
            logging.info("Reading mask from %s", self.mask_file)
            self.mask = daisy.open_ds(self.mask_file, self.mask_dataset,
                                      mode='r')
        else:
            self.mask = None

        delete_ds = False
        if self.overwrite:
            delete_ds = True

        if self.context is None:
            self.context = daisy.Coordinate((0,)*self.affs.roi.dims())
        else:
            self.context = daisy.Coordinate(self.context)

        if self.fragments_in_xy:
            # for CB2
            # if we extract fragments in xy, there is no need to have context in Z
            self.context = [n for n in self.context]
            self.context[0] = 0
            self.context = tuple(self.context)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            # get ROI of source
            assert self.raw_file is not None and self.raw_dataset is not None
            source = daisy.open_ds(self.raw_file, self.raw_dataset)
            dataset_roi = source.roi
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
            total_roi = total_roi.grow(self.context, self.context)

            if self.filedb_roi_offset is None:
                self.filedb_roi_offset = (0, 0, 0)

        else:
            dataset_roi = self.affs.roi
            total_roi = self.affs.roi.grow(self.context, self.context)

            if self.filedb_roi_offset is None:
                self.filedb_roi_offset = dataset_roi.get_begin()

        self.block_size_original = self.block_size
        self.scheduling_chunks = daisy.Coordinate(self.scheduling_chunks)
        self.dataset_chunks = daisy.Coordinate(self.dataset_chunks)

        assert self.database_chunks == [1, 1, 1], "Unsupported for now"
        self.database_chunks = daisy.Coordinate(self.database_chunks)

        dataset_blocksize = daisy.Coordinate(self.block_size)*self.dataset_chunks
        database_blocksize = daisy.Coordinate(self.block_size)*self.database_chunks

        # prepare fragments dataset
        voxel_size = self.affs.voxel_size
        self.fragments_out = daisy.prepare_ds(
            self.fragments_file,
            self.fragments_dataset,
            dataset_roi,
            voxel_size,
            np.uint64,
            # daisy.Roi((0, 0, 0), self.block_size),
            write_size=tuple(dataset_blocksize),
            force_exact_write_size=self.force_exact_write_size,
            compressor={'id': 'zlib', 'level': 3},
            delete=delete_ds,
            )

        if self.db_file is None:
            self.db_file = self.fragments_file

        if self.db_file_name is not None:
            self.rag_provider = daisy.persistence.FileGraphProvider(
                directory=os.path.join(self.db_file, self.db_file_name),
                chunk_size=database_blocksize,
                mode='r+',
                directed=False,
                position_attribute=['center_z', 'center_y', 'center_x'],
                save_attributes_as_single_file=True,
                roi_offset=self.filedb_roi_offset,
                nodes_chunk_size=database_blocksize,
                )
        else:
            assert False, "Deprecated"
            self.rag_provider = daisy.persistence.MongoDbGraphProvider(
                self.db_name,
                host=self.db_host,
                mode='r+',
                directed=False,
                position_attribute=['center_z', 'center_y', 'center_x'],
                indexing_block_size=self.indexing_block_size,
                )

        self.overwrite_mask = None
        if self.overwrite_mask_f:
            # force precheck = False for any ROI with any voxel in mask = 1
            self.overwrite_mask = daisy.open_ds(
                self.overwrite_mask_f, "overwrite_mask")

        if self.overwrite_sections is not None:
            write_shape = [k for k in total_roi.get_shape()]
            write_shape[0] = 40
            write_shape = tuple(write_shape)

            rois = []
            for s in self.overwrite_sections:
                write_offset = [k for k in total_roi.get_begin()]
                write_offset[0] = s*40
                rois.append(daisy.Roi(write_offset, write_shape))

            self.overwrite_sections = rois

        if (self.capillary_pred_file is not None or
                self.capillary_pred_dataset is not None):
            assert self.capillary_pred_file is not None, \
                   "Both capillary_pred_file and " \
                   "capillary_pred_dataset must be defined"
            assert self.capillary_pred_dataset is not None, \
                   "Both capillary_pred_file and " \
                   "capillary_pred_dataset must be defined"

        # adjust min seed distance base on voxel size
        if self.min_seed_distance is None:
            if voxel_size[2] == 8:
                # for 40x8x8
                self.min_seed_distance = 8
            elif voxel_size[2] == 16:
                # for 40x16x16
                self.min_seed_distance = 5
            elif voxel_size[2] == 50:
                # for 50x50x50
                self.min_seed_distance = 5
            else:
                self.min_seed_distance = 10

        config = {
            'affs_file': self.affs_file,
            'affs_dataset': self.affs_dataset,
            'myelin_dataset': 'volumes/myelin',
            'mask_file': self.mask_file,
            'mask_dataset': self.mask_dataset,
            'block_size': self.block_size,
            'block_size_original': self.block_size_original,
            'scheduling_chunks': self.scheduling_chunks,
            'indexing_block_size': self.indexing_block_size,
            'context': self.context,
            'db_host': self.db_host,
            'db_name': self.db_name,
            'num_workers': self.num_workers,
            'fragments_in_xy': self.fragments_in_xy,
            'mask_fragments': self.mask_fragments,
            'fragments_file': self.fragments_file,
            'fragments_dataset': self.fragments_dataset,
            'epsilon_agglomerate': self.epsilon_agglomerate,
            'use_mahotas': self.use_mahotas,
            'min_seed_distance': self.min_seed_distance,
            'capillary_pred_file': self.capillary_pred_file,
            'capillary_pred_dataset': self.capillary_pred_dataset,
            'filter_fragments': self.filter_fragments,
            'db_file': self.db_file,
            'db_file_name': self.db_file_name,
            'filedb_roi_offset': self.filedb_roi_offset,
            'database_blocksize': database_blocksize,
            'block_id_add_one_fix': self.block_id_add_one_fix,
        }

        self.slurmSetup(config, 'actor_fragment_extract.py')

        check_function = (
                lambda b: self.check(b, precheck=True),
                lambda b: self.check(b, precheck=False)
                )
        if self.overwrite:
            check_function = None

        scheduling_block_size = daisy.Coordinate(self.block_size)*self.scheduling_chunks
        read_roi = daisy.Roi((0,)*total_roi.dims(),
                             scheduling_block_size).grow(self.context, self.context)
        write_roi = daisy.Roi((0,)*total_roi.dims(), scheduling_block_size)

        self.schedule(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=self.new_actor,
            check_function=check_function,
            read_write_conflict=False,
            fit='shrink',
            num_workers=self.num_workers)

    def check(self, block, precheck):

        # write_roi = block.write_roi
        # precheck_with_db
        # precheck_use_affs
        if precheck and self.overwrite_sections is not None:
            read_roi_mask = self.overwrite_mask.roi.intersect(block.read_roi)
            for roi in self.overwrite_sections:
                if roi.intersects(read_roi_mask):
                    logger.debug("Block overlaps overwrite_sections %s" % roi)
                    return False

        if precheck and self.overwrite_mask:
            read_roi_mask = self.overwrite_mask.roi.intersect(block.read_roi)
            if not read_roi_mask.empty():
                try:
                    sum = np.sum(self.overwrite_mask[read_roi_mask].to_ndarray())
                    if sum != 0:
                        logger.debug("Block inside overwrite_mask")
                        return False
                except:
                    return False

        if self.precheck_with_db:
            # print("block.write_roi:", block.write_roi)
            center = (block.write_roi.get_begin() + block.write_roi.get_end()) / 2
            # shrinked_roi = shrink_roi(block.write_roi, .95)
            shrinked_roi = daisy.Roi(center, (1, 1, 1))
            # print("shrinked_roi:", shrinked_roi)
            shrinked_roi = shrinked_roi.snap_to_grid(self.block_size_original)
            # print("shrinked_roi:", shrinked_roi)

            if self.rag_provider.num_nodes(shrinked_roi):
                # self.recording_block_done(block)
                return True
            elif self.rag_provider.num_nodes(block.write_roi):
                # just making sure and check the entire block
                return True
            else:
                return False

        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            logger.debug("Skipping block with db check")
            return True

        # # check using rag_provider.num_nodes for compatibility with older runs
        # done = self.rag_provider.num_nodes(block.write_roi) > 0
        # if done:
        #     self.recording_block_done(block)
        #     return True

        return False

    def requires(self):
        if self.no_check_dependency:
            return []
        else:
            return [PredictTask(global_config=self.global_config)]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    if global_config["Input"].get('block_id_add_one_fix', False):
        # fix for cb2_v4 dataset where one (1) was used for the first block id
        # future datasets should just use zero (0)
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True
        global_config["ExtractFragmentTask"]['block_id_add_one_fix'] = True

    req_roi = None
    if "request_offset" in global_config["Input"]:
        req_roi = daisy.Roi(
            tuple(global_config["Input"]["request_offset"]),
            tuple(global_config["Input"]["request_shape"]))
        req_roi = [req_roi]

    daisy.distribute(
        [{'task': ExtractFragmentTask(global_config=global_config,
                                      **user_configs),
         'request': req_roi}],
        global_config=global_config)
