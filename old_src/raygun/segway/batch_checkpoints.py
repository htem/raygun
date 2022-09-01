# import json
import logging
import sys
import glob
import daisy
import copy
sys.path.insert(0, 'segway/tasks')
import task_helper
from task_04_extract_segmentation import SegmentationTask

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(
        sys.argv[1:], aggregate_configs=False)

    orig_global_config = copy.deepcopy(global_config)
    network_conf = global_config["Network"]
    network_path = network_conf["train_dir"]
    checkpoints = glob.glob(
        network_path + "/" + "*.data-00000-of-00001")
    checkpoints = [int(c.split('.')[0].split('_')[-1]) for c in checkpoints]

    if ("batch_min_iteration" in network_conf or
            "batch_max_iteration" in network_conf):
        assert("batch_min_iteration" in network_conf)
        assert("batch_max_iteration" in network_conf)

        checkpoints = [
            c for c in checkpoints if (
                c >= network_conf["batch_min_iteration"] and
                c <= network_conf["batch_max_iteration"])]

    # print(checkpoints); exit(0)

    for c in checkpoints:
        global_config = copy.deepcopy(orig_global_config)
        print("Running inference for iteration %s" % c)
        global_config["Network"]["iteration"] = c
        task_helper.aggregateConfigs(global_config)
        print(global_config)

        daisy.distribute(
            [{'task': SegmentationTask(global_config=global_config,
                                       **user_configs),
             'request': None}],
            global_config=global_config)
