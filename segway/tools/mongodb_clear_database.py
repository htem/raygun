# import json
import logging
import sys

import pymongo

logger = logging.getLogger(__name__)

blacklist_db = [
    '201906_purkinje_cell',
    'IMPORTANT_vnc_t1_vnc_network_540000',
    'synful_1908_vnc1_synapsesV2_setup06_ds4_v3_500000',
]

blacklist_keywords = [
    'xray',
]

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    db_name = sys.argv[1]
    db_host = "134.174.149.150"
    myclient = pymongo.MongoClient(db_host)

    drop_list = [
        '201906_purkinje_cell',
    ]

    for db in drop_list:
        if db in blacklist_db:
            raise RuntimeError("%s is in blacklist_db" % db)
        for kw in blacklist_keywords:
            if kw in db:
                raise RuntimeError("Keyword %s is in %s" % (kw, db))

    for db in drop_list:
        print("Dropping %s..." % db_name)
        # myclient.drop_database(db_name)
