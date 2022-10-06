#%%
from functools import partial
from reloading import reloading
from glob import glob
import json
from subprocess import call
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


def find_source_path(path_template, try_path, n_search=3):
    try_path = try_path.rstrip("/")
    n = -1
    while len(glob(path_template.replace("$source_dirname", try_path + "/*" * n))) == 0:
        if n - 1 > n_search:
            raise ValueError(
                f"Source not found at {path_template.replace('$source_dirname/', try_path + '/*' * n)}"
            )
        try_path = os.path.dirname(try_path)
        n += 1
    source_path = glob(path_template.replace("$source_dirname", try_path + "/*" * n))[0]
    return source_path


def update_validation_configs(config, iter=None):
    config = read_config(config)

    if iter is not None:
        config["checkpoint"] = iter
        config["predict_config"]["checkpoint"] = iter

    train_config = read_config(config["predict_config"]["config_path"])
    sources = train_config["sources"]
    raw_src = sources[np.argmax(["raw" in src.keys() for src in sources])]

    source_path = find_source_path(
        config["predict_config"]["source_path"], raw_src["path"]
    )
    config["predict_config"]["source_path"] = source_path

    source_ds = config["predict_config"]["source_dataset"].replace(
        "$source_dataset", raw_src["raw"]
    )
    config["predict_config"]["source_dataset"] = source_ds

    validation_config_path = config["validation_config_path"]
    to_json(config, validation_config_path)
    prediction_config_path = config["prediction_config_path"]
    to_json(config["predict_config"], prediction_config_path)

    return config


def launch(launch_command):
    try:
        retcode = call(launch_command, shell=True)
        if retcode < 0:
            logger.warning(f"Child was terminated by signal {-retcode}")
        else:
            logger.info(f"Child returned {retcode}")
    except OSError as e:
        logger.warning(f"Execution failed: {e}")


# @reloading
def run_validation(config=None, iter=None):
    if config is None:  # assume used as CLI
        config = sys.argv[1]
        try:
            iter = sys.argv[2]
        except:
            pass
    config = update_validation_configs(config, iter)
    # launch validation
    launch(config["launch_command"])


def validate_affinities(config=None):
    if config is None:
        config = sys.argv[1]

    config = read_config(config)
    # MAKE SURE TO UPDATE CHECKPOINT, "save" in segment config is false
    logger.info("Predicting validation volume affinities...")
    try:  # TODO: Figure out why this is necessary and fix
        predict(config["prediction_config_path"])
    except:
        predict.predict(config["prediction_config_path"])

    if (
        "launch_command" in config["segment_config"].keys()
    ):  # TODO: make this general use in 0.3.0
        launch(config["segment_config"]["launch_command"])
    else:
        validate_segmentation(config)


def validate_segmentation(config=None):
    if config is None:
        config = sys.argv[1]

    config = read_config(config)
    try:  # TODO: Figure out why this is necessary and fix
        seg = segment(config["segment_config"])
    except:
        seg = segment.segment(config["segment_config"])
    image = rasterize_skeleton(config["skeleton_config"])
    logger.info("Evaluating...")
    evaluation = pad_eval(seg, image)
    logger.info("Done... saving...")

    # save metrics
    current_iteration = config["checkpoint"]
    metric_path = config["metric_path"]
    if not os.path.isfile(metric_path):
        metrics = {current_iteration: evaluation}
    else:
        metrics = load_json_file(metric_path)
        metrics[current_iteration] = evaluation
    to_json(metrics, metric_path)
    logger.info("Done.")


def pad_eval(segment_array, image):
    pad = daisy.Coordinate(np.array(image.shape) - np.array(segment_array.shape)) // 2
    if sum(pad) < 3:
        image = image
    else:
        image = image[pad[0] : -pad[0], pad[1] : -pad[1], pad[2] : -pad[2]]

    return rand_voi(image, segment_array)


def evaluate_segmentations(config_path=None):  # TODO: Determine if this is depracated
    if config_path is None:
        config_path = sys.argv[1]

    config = read_config(config_path)
    # Initialize rasterized skeleton image
    image = rasterize_skeleton(config_path)

    # load segmentations
    logger.info(f"Getting segmentation datasets...")
    evaluation = {}
    if "file" in config["eval_sources"].keys():
        segment_file = config["eval_sources"]["file"]
        segment_datasets = config["eval_sources"]["datasets"]
        if isinstance(segment_datasets, str):
            segment_datasets = []
            for ds in glob(
                os.path.join(segment_file, segment_datasets.rstrip("*") + "*")
            ):
                ds_parts = ds.strip("/").split("/")
                ind = (
                    len(ds_parts) - np.nonzero([".n5" in p for p in ds_parts])[0][0] - 1
                )
                segment_datasets.append(os.path.join(ds_parts[-ind:]))

        logger.info(f"Evaluating skeleton...")
        for segment_dataset in segment_datasets:
            segment_ds = daisy.open_ds(segment_file, segment_dataset)
            segment_array = segment_ds.to_ndarray(segment_ds.roi)

            evaluation[segment_dataset] = pad_eval(segment_array, image)

    else:
        for name, dataset in config["eval_sources"]:
            segment_file = dataset["file"]

            logger.info(f"Evaluating {name}...")
            if isinstance(dataset["datasets"], str):
                segment_ds = daisy.open_ds(segment_file, dataset["datasets"])
                segment_array = segment_ds.to_ndarray(segment_ds.roi)

                evaluation[name] = pad_eval(segment_array, image)

            else:
                for segment_dataset in dataset["datasets"]:
                    segment_ds = daisy.open_ds(segment_file, segment_dataset)
                    segment_array = segment_ds.to_ndarray(segment_ds.roi)

                    evaluation[f"{name}_{segment_dataset}"] = pad_eval(
                        segment_array, image
                    )

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
