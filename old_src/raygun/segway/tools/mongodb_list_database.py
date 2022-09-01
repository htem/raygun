# import json
import sys
import pymongo


if __name__ == "__main__":

    try:
        user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])
        db_host = global_config["Input"]["db_host"]
    except:
        db_host = "134.174.149.150"

    myclient = pymongo.MongoClient(db_host)
    print(myclient.database_names())
