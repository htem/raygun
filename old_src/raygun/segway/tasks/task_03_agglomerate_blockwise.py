# import json
import logging
import numpy as np
import sys
import os
# sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/daisy')
import daisy
import task_helper2 as task_helper
from task_02_extract_fragments import ExtractFragmentTask

logger = logging.getLogger(__name__)


def shrink_roi(roi, pct):

    assert pct <= 1.0

    if pct <= 0.01:
        return roi

    check_mult = 2.0/pct
    roi = roi.grow(-roi.get_shape()/check_mult, -roi.get_shape()/check_mult)
    return roi


class AgglomerateTask(task_helper.SlurmTask):
    '''
    Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        in_file (``string``):

            The input file containing affs and fragments.

        affs_dataset, fragments_dataset (``string``):

            Where to find the affinities and fragments.

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

        merge_function (``string``):

            Symbolic name of a merge function. See dictionary below.
    '''

    affs_file = daisy.Parameter()
    affs_dataset = daisy.Parameter()
    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    indexing_block_size = daisy.Parameter(None)
    block_size = daisy.Parameter()
    filedb_nodes_chunk_size = daisy.Parameter(None)
    filedb_edges_chunk_size = daisy.Parameter(None)
    context = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    db_file = daisy.Parameter(None)
    db_file_name = daisy.Parameter("db_file")
    filedb_roi_offset = daisy.Parameter(None)
    filedb_edges_roi_offset = daisy.Parameter(None)
    num_workers = daisy.Parameter()
    merge_function = daisy.Parameter()
    threshold = daisy.Parameter(default=1.0)
    edges_collection = daisy.Parameter() # debug
    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)
    block_id_add_one_fix = daisy.Parameter(False)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        logging.info("Reading affs from %s", self.affs_file)
        affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        logging.info("Reading fragments from %s", self.fragments_file)
        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset, mode='r')

        assert fragments.data.dtype == np.uint64

        # shape = affs.shape[1:]
        self.context = daisy.Coordinate(self.context)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
            total_roi = total_roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*total_roi.dims(),
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*total_roi.dims(), self.block_size)
            if self.filedb_roi_offset is None:
                self.filedb_roi_offset = (0, 0, 0)

        else:
            total_roi = affs.roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*affs.roi.dims(), self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*affs.roi.dims(), self.block_size)
            if self.filedb_roi_offset is None:
                self.filedb_roi_offset = affs.roi.get_begin()

        if self.db_file is None:
            self.db_file = self.fragments_file

        if self.filedb_edges_roi_offset is None:
            self.filedb_edges_roi_offset = self.filedb_roi_offset

        # open RAG DB
        if self.db_file_name is not None:
            # assert self.fragments_block_size is not None
            self.rag_provider = daisy.persistence.FileGraphProvider(
                directory=os.path.join(self.db_file, self.db_file_name),
                chunk_size=None,
                mode='r+',
                directed=False,
                position_attribute=['center_z', 'center_y', 'center_x'],
                save_attributes_as_single_file=True,
                roi_offset=self.filedb_roi_offset,
                nodes_chunk_size=self.filedb_nodes_chunk_size,
                edges_chunk_size=self.filedb_edges_chunk_size,
                edges_roi_offset=self.filedb_edges_roi_offset,
                )
        else:
            assert False, "Deprecated"
            self.rag_provider = daisy.persistence.MongoDbGraphProvider(
                self.db_name,
                host=self.db_host,
                mode='r+',
                directed=False,
                edges_collection='edges_' + self.merge_function,
                position_attribute=['center_z', 'center_y', 'center_x'],
                indexing_block_size=self.indexing_block_size,
            )

        config = {
            'affs_file': self.affs_file,
            'affs_dataset': self.affs_dataset,
            'fragments_file': self.fragments_file,
            'fragments_dataset': self.fragments_dataset,
            'block_size': self.block_size,
            'context': self.context,
            'db_host': self.db_host,
            'db_name': self.db_name,
            'num_workers': self.num_workers,
            'merge_function': self.merge_function,
            'threshold': self.threshold,
            'indexing_block_size': self.indexing_block_size,
            'db_file': self.db_file,
            'db_file_name': self.db_file_name,
            'filedb_nodes_chunk_size': self.filedb_nodes_chunk_size,
            'filedb_edges_chunk_size': self.filedb_edges_chunk_size,
            'filedb_roi_offset': self.filedb_roi_offset,
            'filedb_edges_roi_offset': self.filedb_edges_roi_offset,
            'block_id_add_one_fix': self.block_id_add_one_fix,
        }
        self.slurmSetup(config, 'actor_agglomerate.py')

        check_function = (self.block_done, lambda b: True)
        if self.overwrite:
            check_function = None

        self.schedule(
            total_roi,
            read_roi,
            write_roi,
            process_function=self.new_actor,
            check_function=check_function,
            num_workers=self.num_workers,
            read_write_conflict=False,
            max_retries=self.max_retries,
            fit='shrink')

    def block_done(self, block):

        db_count = self.completion_db.count({'block_id': block.block_id})
        if db_count == 0:
            return False

        # checking with DB is somehow not reliable
        # if db_count:
        #     logger.debug("Skipping block with db check")
        #     return True

        if self.rag_provider.has_edges(block.write_roi):
            return True

        # no nodes found, means an error in fragment extract; skip
        return False

    def requires(self):
        if self.no_check_dependency:
            return []
        return [ExtractFragmentTask(global_config=self.global_config)]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    if global_config["Input"].get('block_id_add_one_fix', False):
        # fix for cb2_v4 dataset where one (1) was used for the first block id
        # future datasets should just use zero (0)
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True
        global_config["AgglomerateTask"]['block_id_add_one_fix'] = True

    req_roi = None
    if "request_offset" in global_config["Input"]:
        req_roi = daisy.Roi(
            tuple(global_config["Input"]["request_offset"]),
            tuple(global_config["Input"]["request_shape"]))
        req_roi = [req_roi]

    daisy.distribute(
        [{'task': AgglomerateTask(global_config=global_config,
                                  **user_configs),
         'request': req_roi}],
        global_config=global_config)
