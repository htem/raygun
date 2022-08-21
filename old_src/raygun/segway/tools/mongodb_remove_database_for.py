# import json
import logging
import sys

import segway.tasks.task_helper2 as task_helper

import pymongo

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    db_host = global_config["Input"]["db_host"]
    myclient = pymongo.MongoClient(db_host)

    db_name = global_config["Input"]["db_name"]
    print("Dropping %s..." % db_name)
    i = input("Sure? Yes/[No] ")
    if i == "Yes":
        myclient.drop_database(db_name)
        print("Dropped %s!" % db_name)
    else:
        print("Aborted")
