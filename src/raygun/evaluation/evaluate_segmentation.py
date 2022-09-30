#%%
from functools import partial
from glob import glob
import json
import daisy
import sys
import os
import json
import zarr
import numpy as np
from funlib.evaluate import rand_voi
from raygun.evaluation.skeleton import rasterize_skeleton
from raygun import predict, read_config, segment

import logging

logger = logging.getLogger(__name__)


def validate_affinities(config):
    config = read_config(config)
    # MAKE SURE TO UPDATE CHECKPOINT, "save" in segment config is false
    logger.info("Predicting validation volume affinities...")
    predict(config["predict_config"])
    return validate_segmentation(config)


def validate_segmentation(config):
    config = read_config(config)
    seg = segment(config["segment_config"])
    image = rasterize_skeleton(config["skeleton_config"])
    pad = daisy.Coordinate(np.array(image.shape) - np.array(seg.shape)) // 2
    if sum(pad) < 3:
        image = image
    else:
        image = image[pad[0] : -pad[0], pad[1] : -pad[1], pad[2] : -pad[2]]
    return rand_voi(image, seg)


def evaluate_segmentations(config_path):
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
    config_file = sys.argv[1]
    thresh_list = "volumes/segmentation_*"
    update_best = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "update_best":
            update_best = True
            increment = (
                config_file.strip("/")
                .split("/")[-1]
                .replace("segment_", "")
                .replace(".json", "")
            )
        else:
            increment = int(sys.argv[2])
    else:
        increment = (
            config_file.strip("/")
            .split("/")[-1]
            .replace("segment_", "")
            .replace(".json", "")
        )
        thresh_list = False

    METRIC_OUT_JSON = "./metrics/metrics.json"
    BEST_METRIC_JSON = "./metrics/best.iteration"

    config = read_config(config_file)
    current_iteration = int(config["Network"]["iteration"])
    print(f"Evaluating {config_file} at iteration {current_iteration}...")
    evaluation = evaluate(config)
    best_eval = {}
    for thresh, metrics in evaluation.items():
        if len(best_eval) == 0 or get_score(best_eval) > get_score(metrics):
            best_eval = metrics.copy()
            best_eval["segment_ds"] = thresh
            best_eval["iteration"] = current_iteration

    # check append
    if not os.path.isfile(METRIC_OUT_JSON):
        metrics = {current_iteration: evaluation}
    else:
        with open(METRIC_OUT_JSON, "r") as f:
            metrics = json.load(f)
        if (
            isinstance(increment, str) and not update_best
        ):  # for evaluating best threshold/iteration on different raw_datasets
            # best_eval[current_iteration]['iteration'] = current_iteration
            # metrics[increment] = best_eval[current_iteration]
            evaluation["iteration"] = current_iteration
            metrics[increment] = evaluation
        else:
            metrics[current_iteration] = evaluation
    with open(METRIC_OUT_JSON, "w") as f:
        json.dump(metrics, f, indent=4)

    # Increment config
    if update_best:
        print(f"New best = {best_eval}")
        with open(BEST_METRIC_JSON, "w") as f:
            json.dump(best_eval, f, indent=4)
    elif increment is not None and not isinstance(increment, str):
        # Save best
        if not os.path.isfile(BEST_METRIC_JSON):
            with open(BEST_METRIC_JSON, "w") as f:
                json.dump(best_eval, f, indent=4)
        else:
            with open(BEST_METRIC_JSON, "r") as f:
                curr_best = json.load(f)

            if get_score(curr_best) > get_score(best_eval):
                print(f"New best = {best_eval}")
                with open(BEST_METRIC_JSON, "w") as f:
                    json.dump(best_eval, f, indent=4)

        # print(config_file)
        with open(config_file, "r+") as f:
            config = json.loads(f.read())

        # Save config file
        with open(config_file, "w+") as f:
            config["Network"]["iteration"] = current_iteration + increment
            json.dump(config, f, indent=4)

# %%
