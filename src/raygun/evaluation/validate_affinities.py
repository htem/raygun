#%%
from functools import partial
from glob import glob
import json
from wsgiref import validate
import daisy
import sys
import os
import json
from raygun.utils import load_json_file, to_json
import zarr
import numpy as np
from funlib.evaluate import rand_voi
from raygun.evaluation.skeleton import rasterize_skeleton
from raygun import predict, read_config, segment

import logging

logger = logging.getLogger(__name__)


def validate_affinities(config=None):
    if config is None:
        config = sys.argv[1]

    config = read_config(config)
    # MAKE SURE TO UPDATE CHECKPOINT, "save" in segment config is false
    logger.info("Predicting validation volume affinities...")
    predict(config["prediction_config_path"])
    return validate_segmentation(config)


def validate_segmentation(config=None):
    if config is None:
        config = sys.argv[1]

    config = read_config(config)
    seg = segment(config["segment_config"])
    image = rasterize_skeleton(config["skeleton_config"])
    pad = daisy.Coordinate(np.array(image.shape) - np.array(seg.shape)) // 2
    if sum(pad) < 3:
        image = image
    else:
        image = image[pad[0] : -pad[0], pad[1] : -pad[1], pad[2] : -pad[2]]
    evaluation = rand_voi(image, seg)

    # save metrics
    current_iteration = config["checkpoint"]
    metric_path = config["metric_path"]
    if not os.path.isfile(metric_path):
        metrics = {current_iteration: evaluation}
    else:
        metrics = load_json_file(metric_path)
        metrics[current_iteration] = evaluation
    to_json(metrics, metric_path)

    return metrics


def evaluate_segmentations(config_path=None):
    if config_path is None:
        config_path = sys.argv[1]

    config = read_config(config_path)
    # Initialize rasterized skeleton image
    image = rasterize_skeleton(config_path)

    # load segmentations
    logger.info(f"Getting segmentation datasets...")
    segment_file = config["eval_sources"]["file"]
    segment_datasets = config["eval_sources"]["datasets"]
    if isinstance(segment_datasets, str):
        segment_datasets = []
        for ds in glob(os.path.join(segment_file, segment_datasets.rstrip("*") + "*")):
            ds_parts = ds.strip("/").split("/")
            ind = len(ds_parts) - np.nonzero([".n5" in p for p in ds_parts])[0][0] - 1
            segment_datasets.append(os.path.join(ds_parts[-ind:]))

    logger.info(f"Evaluating skeleton...")
    evaluation = {}
    for segment_dataset in segment_datasets:
        segment_ds = daisy.open_ds(segment_file, segment_dataset)
        segment_array = segment_ds.to_ndarray(segment_ds.roi)
        pad = (
            daisy.Coordinate(np.array(image.shape) - np.array(segment_array.shape)) // 2
        )
        if sum(pad) < 3:
            this_image = image
        else:
            this_image = image[pad[0] : -pad[0], pad[1] : -pad[1], pad[2] : -pad[2]]

        evaluation[segment_dataset] = rand_voi(this_image, segment_array)

    return evaluation


def get_score(metrics, keys=["nvi_split", "nvi_merge"]):
    score = 0
    for key in keys:
        if not np.isnan(metrics[key]):
            score += metrics[key]
            # if metrics[key] != 0: #Discard any 0 metrics as flawed(?)
            #     score *= metrics[key]
        else:
            return 999
    return score


#%%
if __name__ == "__main__":
    validate_affinities()

# %%
