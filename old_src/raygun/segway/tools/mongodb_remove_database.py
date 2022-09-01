# import json
import logging
import sys

import pymongo

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    db_name = sys.argv[1]
    # db_host = "134.174.149.150"
    db_host = "mongodb://10.117.28.250:27018"
    myclient = pymongo.MongoClient(db_host)

    print("Dropping %s..." % db_name)
    i = input("Sure? Yes/[No] ")
    if i == "Yes":
        myclient.drop_database(db_name)
        print("Dropped %s!" % db_name)
    else:
        print("Aborted")
