import json
import os
import logging
import numpy as np
import sys

import daisy
from lsd.parallel_fragments import watershed_in_block
import pymongo

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    mask_fragments = False
    mask_file = None
    mask_dataset = None
    epsilon_agglomerate = 0
    min_seed_distance = 10  # default seed size from Jan
    capillary_pred_file = None
    capillary_pred_dataset = None

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    if run_config.get('block_id_add_one_fix', False):
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    if mask_fragments:
        raise Exception("Not tested")
        logging.info("Reading mask from %s", mask_file)
        mask = daisy.open_ds(mask_file, mask_dataset, mode='r')
    else:
        mask = None

    fragments_out = daisy.open_ds(
        fragments_file, fragments_dataset, 'r+')

    try:
        myelin_ds = daisy.open_ds(
            fragments_file, myelin_dataset)
    except:
        myelin_ds = None

    filter_masks = []
    print(capillary_pred_file)
    print(capillary_pred_dataset)

    if capillary_pred_file is not None and capillary_pred_dataset is not None:
        capillary_pred_ds = daisy.open_ds(capillary_pred_file, capillary_pred_dataset)
        filter_masks.append(capillary_pred_ds)

    # open RAG DB
    if db_file_name is not None:
        rag_provider = daisy.persistence.FileGraphProvider(
            directory=os.path.join(db_file, db_file_name),
            chunk_size=database_blocksize,
            mode='r+',
            directed=False,
            position_attribute=['center_z', 'center_y', 'center_x'],
            save_attributes_as_single_file=True,
            roi_offset=filedb_roi_offset,
            nodes_chunk_size=database_blocksize,
            )
    else:
        rag_provider = daisy.persistence.MongoDbGraphProvider(
            db_name,
            host=db_host,
            mode='r+',
            directed=False,
            position_attribute=['center_z', 'center_y', 'center_x'],
            indexing_block_size=indexing_block_size,
            )

    assert fragments_out.data.dtype == np.uint64

    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        logging.info("Running fragment extraction for block %s" % block)

        write_roi = block.write_roi
        # read_roi = block.read_roi

        fragments_out_buffer = daisy.Array(
            np.zeros(
                write_roi.get_shape()/fragments_out.voxel_size,
                fragments_out.dtype),
            write_roi,
            fragments_out.voxel_size)

        for chunk in daisy.Block.get_chunks(
                block,
                chunk_div=None,
                chunk_shape=block_size_original
                ):

            logging.info("Chunk %s" % chunk)

            watershed_in_block(affs,
                               chunk,
                               rag_provider,
                               fragments_out_buffer,
                               fragments_in_xy,
                               epsilon_agglomerate,
                               mask,
                               filter_fragments=filter_fragments,
                               min_seed_distance=min_seed_distance,
                               filter_masks=filter_masks,
                               )

        print("write_roi:", write_roi)
        fragments_out[write_roi] = fragments_out_buffer[write_roi]

        # recording block done in the database
        document = dict()
        document.update({
            'block_id': block.block_id,
        })
        completion_db.insert(document)

        client_scheduler.release_block(block, ret=0)
