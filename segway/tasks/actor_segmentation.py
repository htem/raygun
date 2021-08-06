import json
import logging
import lsd
import os
import daisy
import sys
import datetime
import copy
from parallel_read_rag import parallel_read_rag

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.mongodb_rag_provider').setLevel(logging.DEBUG)


def segment(
        fragments_file,
        fragments_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        edges_collection,
        thresholds,
        block=None,
        num_workers=4):

    logging.info(
        "%s: segmentation started" % datetime.datetime.now()
        )

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    # open RAG DB
    # rag_provider = lsd.persistence.MongoDbRagProvider(
    #     db_name,
    #     host=db_host,
    #     mode='r',
    #     edges_collection=edges_collection)

    if block is not None:
        total_roi = block.write_roi
    else:
        total_roi = fragments.roi
        assert(0)
        # if roi_offset is not None:
        #     assert roi_shape is not None, "If roi_offset is set, roi_shape " \
        #                                   "also needs to be provided"
        #     total_roi = daisy.Roi(offset=roi_offset, shape=roi_shape)

    # slice
    logging.info("Reading fragments and RAG in %s" % total_roi)
    fragments = fragments[total_roi]
    # rag = parallel_read_rag(
    #     total_roi,
    #     db_host,
    #     db_name,
    #     edges_collection=edges_collection,
    #     block_size=(4096, 4096, 4096),
    #     num_workers=num_workers,
    #     retry=0)
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='r+',
        edges_collection=edges_collection)
    rag = rag_provider[total_roi]

    logging.info(
        "%s: finished reading DB" % datetime.datetime.now()
        )
    logging.info("Number of nodes in RAG: %d" % (len(rag.nodes())))
    logging.info("Number of edges in RAG: %d" % (len(rag.edges())))

    logging.info(
        "%s: loaded to memory" % datetime.datetime.now()
        )
    # create a segmentation
    logging.info("Merging...")
    segments = fragments.to_ndarray()
    out_dataset_base = out_dataset

    for threshold in thresholds:

        segmentation_data = copy.deepcopy(segments)

        rag.get_segmentation(threshold, segmentation_data)

        logging.info(
            "%s: merged" % datetime.datetime.now()
            )
        # store segmentation
        logging.info("Writing segmentation for threshold %f..." % threshold)
        out_dataset = out_dataset_base + "_%.3f" % threshold
        segmentation = daisy.prepare_ds(
            out_file,
            out_dataset,
            fragments.roi,
            fragments.voxel_size,
            fragments.data.dtype,
            # temporary fix until
            # https://github.com/zarr-developers/numcodecs/pull/87 gets
            # approved
            # (we want gzip to be the default)
            compressor={'id': 'zlib', 'level': 5})
        segmentation.data[:] = segmentation_data


if __name__ == "__main__":
    logging.info(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    fragments_file = None
    fragments_dataset = None
    out_file = None
    out_dataset = None
    db_host = None
    db_name = None
    edges_collection = None
    thresholds = None
    num_workers = 4

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    logging.info("WORKER: Running with context %s" % os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    while True:

        block = client_scheduler.acquire_block()
        if block is None:
            break

        segment(
            fragments_file,
            fragments_dataset,
            out_file,
            out_dataset,
            db_host,
            db_name,
            edges_collection,
            thresholds,
            block,
            num_workers)

        client_scheduler.release_block(block, ret=0)
