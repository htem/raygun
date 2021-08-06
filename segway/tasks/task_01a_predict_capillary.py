import logging
import sys
import daisy
import task_helper2 as task_helper

from task_predict_ilastik import PredictIlastikTask

logger = logging.getLogger(__name__)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    if global_config["Input"].get('block_id_add_one_fix', False):
        # fix for cb2_v4 dataset where one (1) was used for the first block id
        # future datasets should just use zero (0)
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True
        global_config["PredictCapillaryTask"]['block_id_add_one_fix'] = True

    req_roi = None
    if "request_offset" in global_config["Input"]:
        req_roi = daisy.Roi(
            tuple(global_config["Input"]["request_offset"]),
            tuple(global_config["Input"]["request_shape"]))
        req_roi = [req_roi]

    daisy.distribute(
        [{'task': PredictIlastikTask(
                            task_id="PredictCapillaryTask",
                            global_config=global_config,
                            **user_configs),
         'request': req_roi}],
        global_config=global_config)
