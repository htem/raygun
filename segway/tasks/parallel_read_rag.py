import multiprocessing
import daisy
import lsd
import logging

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def read_graph_in_block(block, rag_provider, shared_node_list, shared_edge_list):

    logger.info("Reading RAG in %s...", block.read_roi)

    # print(rag_provider)
    rag = rag_provider[block.read_roi]
    # print(rag.nodes)
    # print(rag)

    # shared_node_list += list(rag.nodes().items())
    shared_node_list += list(rag.nodes())
    # print(rag.nodes())
    # print(rag.edges(data=True))
    shared_edge_list += [
        (e0, e1, data)
        # for e, data in rag.edges.items()
        for e0, e1, data in rag.edges(data=True)
    ]

def parallel_read_rag(
        roi,
        db_host,
        db_name,
        edges_collection,
        block_size,
        num_workers,
        retry=0):

    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name=db_name,
        host=db_host,
        edges_collection=edges_collection,
        mode='r')
    print(rag_provider)
    node_list = multiprocessing.Manager().list()
    edge_list = multiprocessing.Manager().list()

    for i in range(retry + 1):
        if daisy.run_blockwise(
            roi,
            read_roi,
            write_roi,
            lambda b: read_graph_in_block(
                b,
                rag_provider,
                node_list,
                edge_list),
            fit='shrink',
            num_workers=num_workers,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel read failed, retrying %d/%d", i + 1, retry)

    graph = lsd.persistence.mongodb_rag_provider.MongoDbSubRag(db_name, db_host, 'r')
    graph.add_nodes_from(node_list)
    graph.add_edges_from(edge_list)
    return graph
