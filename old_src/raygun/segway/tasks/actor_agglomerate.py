import json
import os
import logging
import numpy as np
import sys

# sys.path.insert(0, '/home/tmn7/segmentation/tri/daisy')
import daisy
# import lsd
from lsd.parallel_aff_agglomerate import agglomerate_in_block
import pymongo

logging.basicConfig(level=logging.INFO)

# logging.getLogger('daisy.persistence.file_graph_provider').setLevel(logging.DEBUG)

if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    if run_config.get('block_id_add_one_fix', False):
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

    waterz_merge_function = {
        'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
        'hist_quant_20': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 20, ScoreValue, 256, false>>',
        'hist_quant_30': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 30, ScoreValue, 256, false>>',
        'hist_quant_40': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 40, ScoreValue, 256, false>>',
        'hist_quant_60': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 60, ScoreValue, 256, false>>',
        'hist_quant_70': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 70, ScoreValue, 256, false>>',
        'hist_quant_80': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 80, ScoreValue, 256, false>>',
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

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    logging.info("Reading fragments from %s", fragments_file)
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

    # open RAG DB
    if db_file_name is not None:
        rag_provider = daisy.persistence.FileGraphProvider(
            directory=os.path.join(db_file, db_file_name),
            chunk_size=None,
            mode='r+',
            directed=False,
            position_attribute=['center_z', 'center_y', 'center_x'],
            save_attributes_as_single_file=True,
            roi_offset=filedb_roi_offset,
            nodes_chunk_size=filedb_nodes_chunk_size,
            edges_chunk_size=filedb_edges_chunk_size,
            edges_roi_offset=filedb_edges_roi_offset,
            )
    else:
        assert False, "Deprecated"
        rag_provider = daisy.persistence.MongoDbGraphProvider(
            db_name,
            host=db_host,
            mode='r+',
            directed=False,
            edges_collection='edges_' + merge_function,
            position_attribute=['center_z', 'center_y', 'center_x'],
            indexing_block_size=indexing_block_size,
            )

    assert fragments.data.dtype == np.uint64

    shape = affs.shape[1:]
    context = daisy.Coordinate(context)

    total_roi = affs.roi.grow(context, context)
    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        logging.info("Running agglomeration for block %s" % block)

        agglomerate_in_block(
                affs,
                fragments,
                rag_provider,
                block,
                waterz_merge_function,
                threshold),

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