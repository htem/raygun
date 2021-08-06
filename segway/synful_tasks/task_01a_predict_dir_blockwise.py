# from funlib.run import run
# import hashlib
import json
import logging
import os
import sys
# import time
# import glob

import daisy
# import numpy as np
# import pymongo

from task_01_predict_blockwise import PredictSynapseTask
from segway.tasks import task_helper2 as task_helper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.blocks').setLevel(logging.DEBUG)


class PredictSynapseDirTask(PredictSynapseTask):

    serialize_gpu_predictions = daisy.Parameter(True)

    def requires(self):
        # if self.serialize_gpu_predictions:
        #     return [PredictSynapseTask(global_config=self.global_config)]
        # else:
            return []


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    req_roi = None
    if "request_offset" in global_config["Input"]:
        req_roi = daisy.Roi(
            tuple(global_config["Input"]["request_offset"]),
            tuple(global_config["Input"]["request_shape"]))
        req_roi = [req_roi]

    daisy.distribute(
        [{'task': PredictSynapseDirTask(global_config=global_config,
                              **user_configs),
         'request': req_roi}],
        global_config=global_config)
