import collections
import logging
import lsd
# import numpy as np
import daisy
from daisy import Coordinate, Roi
import sys
import numpy as np

import task_helper
from task_sparse_segmentation import GrowSegmentationTask


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_args, global_config = task_helper.parseConfigs(sys.argv[1:])
    config = global_config["SparseSegmentationServer"]

    for k in user_args:
        config[k] = int(user_args[k])

    print(config)

    block_size = config['block_size']
    iterations = config['iterations']
    reinitialize = config['reinitialize']
    segment_id = config['segment_id']
    seed = config['seeds'][0]
    continuous = config['continuous']

    logging.info("Reading fragments from %s", config['fragments_file'])
    fragments = daisy.open_ds(config['fragments_file'],
                              config['fragments_dataset'],
                              mode='r')

    logging.info("Opening RAG DB...")
    rag_provider = lsd.persistence.MongoDbRagProvider(
        config["db_name"],
        host=config["db_host"],
        mode='r+',
        edges_collection=config["edges_collection"])

    # generate segment ID

    # generate session_hash
    session_hash = "deadbeef"

    db = rag_provider.get_db()

    grow_collection = "task_grow_segment_%s" % session_hash
    skipped_collection = "task_grow_skipped_%s" % session_hash

    if reinitialize:
        logger.info("Reinitialize with seed")
        db.drop_collection(grow_collection)
        db.drop_collection(skipped_collection)

    # process seeds
    write_roi = daisy.Roi((0, 0, 0), block_size)
    request_roi = daisy.expand_write_roi_to_grid(
            Roi(tuple(seed), (1, 1, 1)),
            write_roi)

    daisy.distribute(
        [{'task': GrowSegmentationTask(
            global_config=global_config,
            segment_id=segment_id,
            seed_zyxs=[seed],
            # session_hash=session_hash
            grow_db_collection=grow_collection,
            skipped_db_collection=skipped_collection
            ),
          'request': [request_roi]}])

    logging.info("Reading segments from %s", config['segment_file'])
    # segment_ds = daisy.open_ds(config['segment_file'],
    #                            config['segment_dataset'],
    #                            mode='r')

    grow_db = db[grow_collection]
    # segment_ds = segment_ds[segment_ds.roi]

    # HACK: until task dependency failure is implemented
    # which can trigger cleanup procedures because we want to remove
    # continuation nodes that cannot be computed
    total_roi = fragments.roi
    total_roi = total_roi.grow(-Coordinate(block_size),
                               -Coordinate(block_size))
    total_roi = total_roi.grow(-Coordinate(block_size),
                               -Coordinate(block_size))

    print(total_roi)

    i = 0
    while i < iterations or continuous:
        logging.info("Running iteration %d" % i)
        db_nodes = [n for n in grow_db.find()]
        # print(db_nodes)

        to_segment_nodes = []
        skipped = []
        for n in db_nodes:
            if total_roi.contains(Coordinate(n["zyx"])):
                to_segment_nodes.append(n)
            else:
                skipped.append(n)

        logger.info("Removing %s", skipped)
        grow_db.remove(
            {'_id': {'$in': [n['_id'] for n in skipped]}})

        if len(to_segment_nodes) == 0:
            break
        # # filter out segmented nodes
        # segmented_nodes = []
        # to_segment_nodes = []
        # for n in db_nodes:
        #     print(n)
        #     if segment_ds[Coordinate(n["zyx"])] != segment_id:
        #         to_segment_nodes.append(n)
        #     else:
        #         segmented_nodes.append(n)

        # logging.info("Unsegmented nodes: %s" %
        #              [n["zyx"] for n in to_segment_nodes])
        # logging.info("Already segmented nodes: %s" %
        #              [n["zyx"] for n in segmented_nodes])

        grow_db.remove(
            {'_id': {'$in': [n['_id'] for n in skipped]}})

        # aggregate nodes into blocks by coordinates
        blocks = collections.defaultdict(list)
        # for node in to_segment_nodes:
        for node in to_segment_nodes:
            grid_zyx = (Coordinate(node["zyx"]) / Coordinate(block_size) *
                        Coordinate(block_size))
            blocks[grid_zyx].append(node)

        logging.info(blocks)

        tasks = []
        for i, block in enumerate(blocks):
            logger.info("Running segmentation for block")
            logger.info(block)
            request_roi = daisy.Roi(block, block_size)
            logger.info(request_roi)

            # make one task per block
            task = {
                'task': GrowSegmentationTask(
                    task_id="GrowSegmentationTask_%d" % i,
                    global_config=global_config,
                    segment_id=segment_id,
                    seed_zyxs=[n["zyx"] for n in blocks[block]],
                    seed_db_ids=[str(n["_id"]) for n in blocks[block]],
                    # session_hash=session_hash
                    grow_db_collection=grow_collection,
                    skipped_db_collection=skipped_collection
                    ),
                'request': [request_roi]}

            tasks.append(task)

            if len(tasks) >= 16:
                # run at most 16 concurrent tasks
                daisy.distribute(tasks)
                tasks = []

        if len(tasks):
            daisy.distribute(tasks)

        i += 1

    rag_provider.disconnect()
