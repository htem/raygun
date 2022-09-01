# import json
import logging
import sys
import os
import os.path as path

# import numpy as np

import daisy

import task_helper2 as task_helper
from task_03_agglomerate_blockwise import AgglomerateTask

from filedb_graph import FileDBGraph

logger = logging.getLogger(__name__)


class MakeInterThresholdMappingTask(task_helper.SlurmTask):

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    out_file = daisy.Parameter(None)
    merge_function = daisy.Parameter()
    edges_collection = daisy.Parameter()
    source_threshold = daisy.Parameter()
    thresholds = daisy.Parameter()
    # num_workers = daisy.Parameter()
    lut_dir = daisy.Parameter(None)

    sub_roi_offset = daisy.Parameter(None)
    sub_roi_shape = daisy.Parameter(None)

    block_size = daisy.Parameter()
    super_chunk_size = daisy.Parameter()
    file_graph_roi_offset = daisy.Parameter()

    # ignore_degenerates = daisy.Parameter(False)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        super_block_size = (
            daisy.Coordinate(self.block_size) * tuple(self.super_chunk_size))

        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
        else:
            total_roi = fragments.roi

        assert fragments.roi.contains(total_roi)

        read_roi = daisy.Roi((0,)*total_roi.dims(), super_block_size)
        write_roi = read_roi

        if self.out_file is None:
            self.out_file = self.fragments_file

        local_graph_dir = os.path.join(
            self.out_file,
            self.lut_dir,)

        super_lut_dir = 'super_%dx%dx%d_%s_%d' % (
            self.super_chunk_size[0], self.super_chunk_size[1], self.super_chunk_size[2],
            self.merge_function,
            self.source_threshold*100)
        subseg_graph_dir = os.path.join(local_graph_dir, super_lut_dir)
        self.subseg_graph = FileDBGraph(
            filepath=subseg_graph_dir,
            blocksize=super_block_size,
            roi_offset=self.file_graph_roi_offset,
            )
        self.subseg_graph.create_attributes("threshold_map", thresholds=self.thresholds)
        self.subseg_graph.create_attributes("threshold_map_debug", thresholds=self.thresholds)

        self.last_threshold = self.thresholds[-1]

        # for threshold in self.thresholds:
        #     os.makedirs(os.path.join(self.out_dir, "threshold_map_%s_%d_%d" % (self.merge_function, int(self.source_threshold*100), int(threshold*100))),
        #         exist_ok=True)
        #     os.makedirs(os.path.join(self.out_dir, "threshold_map_%s_%d_%d/debug" % (self.merge_function, int(self.source_threshold*100), int(threshold*100))),
        #         exist_ok=True)

        config = {
            'db_host': self.db_host,
            'db_name': self.db_name,
            'fragments_file': self.fragments_file,
            'local_graph_dir': local_graph_dir,
            'subseg_graph_dir': subseg_graph_dir,
            'local_block_size': self.block_size,
            'super_block_size': super_block_size,
            'file_graph_roi_offset': self.file_graph_roi_offset,
            'merge_function': self.merge_function,
            # 'edges_collection': self.edges_collection,
            'thresholds': self.thresholds,
            'source_threshold': self.source_threshold,
            # 'ignore_degenerates': self.ignore_degenerates,
            # 'db_file': self.db_file,
            # 'db_file_name': self.db_file_name,
            # 'filedb_nodes_chunk_size': self.filedb_nodes_chunk_size,
            # 'filedb_edges_chunk_size': self.filedb_edges_chunk_size,
            # 'filedb_roi_offset': self.filedb_roi_offset,
            # 'filedb_edges_roi_offset': self.filedb_edges_roi_offset,
        }
        self.slurmSetup(
            config,
            '04aa_find_segments_blockwise.py')

        check_function = self.block_done
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

    def requires(self):
        if self.no_check_dependency:
            return []
        return [AgglomerateTask(global_config=self.global_config)]

    def block_done(self, block):

        # if self.completion_db.count({'block_id': block.block_id}) >= 1:
        #     logger.debug("Skipping block with db check")
        #     return True

        # block_id = block.block_id
        # threshold = self.thresholds[-1]
        # # lookup = "threshold_map_%s_%d_%d/%d.npz" % (
        # #     self.merge_function, int(self.source_threshold*100),
        # #     int(threshold*100), block_id)
        # path = get_subseg_attribute_path("threshold_map_%d_%d" % (self.source_threshold, threshold))
        # out_file = os.path.join(path, "%d.npz")
        # # print(lookup)
        # # out_file = os.path.join(self.out_dir, lookup)
        # # logger.info("Checking %s" % out_file)
        # exists = path.exists(out_file)

        exists = self.subseg_graph.exists_attribute(
            attr="threshold_map",
            block=block,
            threshold=self.last_threshold,
            )

        return exists


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    if global_config["Input"].get('block_id_add_one_fix', False):
        # fix for cb2_v4 dataset where one (1) was used for the first block id
        # future datasets should just use zero (0)
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

    daisy.distribute(
        [{'task': MakeInterThresholdMappingTask(global_config=global_config,
                                            **user_configs),
         'request': None}],
        global_config=global_config)
