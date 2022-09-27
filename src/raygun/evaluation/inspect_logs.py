#%%
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
from raygun import read_config
import tensorflow as tf
from glob import glob
import os
import sys

from raygun.utils import to_json


def parse_events_file(path: str, tags: list):
    metrics = defaultdict(list)
    for e in tf.compat.v1.train.summary_iterator(path):
        for v in e.summary.value:
            if v.tag in tags:
                if v.tag == tags[0]:
                    metrics["step"].append(e.step)
                metrics[v.tag].append(v.simple_value)
    for k, v in metrics.items():
        metrics[k] = np.array(v)
    return metrics


# %%
def pick_checkpoints(
    meta_log_dir="/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN_7/tensorboards",
    increment=2000,
    start=2000,  # TODO: ALLOW FOR UNBOUNDED
    final=200000,
    tags=["l1_loss/cycled_A", "l1_loss/cycled_B", "gan_loss/fake_A", "gan_loss/fake_B"],
    smoothing=0.999,
    plot=True,
    save=False,
):
    meta_log_dir = meta_log_dir.rstrip("/*") + "/*"

    folders = glob(meta_log_dir)
    model_logs = {}  # model_name: log_metrics
    for folder in folders:
        log_paths = glob(folder + "/*")
        log_path = max(log_paths, key=os.path.getctime)
        model_name = folder.split("/")[-1]
        model_logs[model_name] = parse_events_file(log_path, tags)
        # check what we want is there:
        p = 0
        while start not in model_logs[model_name]["step"] and p < len(log_paths):
            model_logs[model_name] = parse_events_file(log_paths[p], tags)
            p += 1

        model_logs[model_name]["geo_mean"] = get_geo_mean(model_logs[model_name], tags)
        # model_logs[model_name]['smooth_geo_mean'] = smooth(model_logs[model_name]['geo_mean'], smoothing)
        model_logs[model_name]["smooth_geo_mean"] = get_geo_mean(
            model_logs[model_name], tags, smoothing=smoothing
        )

    for model_name in model_logs.keys():
        inds = np.array(
            [
                np.where(model_logs[model_name]["step"] == step)
                for step in np.arange(start, final + increment, increment)
            ]
        ).flatten()
        model_logs[model_name]["score_steps"] = np.arange(
            start, final + increment, increment
        )
        model_logs[model_name]["scores"] = model_logs[model_name]["smooth_geo_mean"][
            inds
        ]
        model_logs[model_name]["best_step"] = model_logs[model_name]["score_steps"][
            model_logs[model_name]["scores"].argmin()
        ]
        for tag in tags + ["geo_mean"]:
            model_logs[model_name][tag] = model_logs[model_name][tag][inds]

    bests = show_best_steps(model_logs)
    if plot:
        plot_all(model_logs, tags + ["scores"])

    if save:
        file_basename = os.path.join(
            os.path.dirname(os.path.dirname(meta_log_dir)), "model_logs"
        )
        to_json(model_logs, file_basename + ".json")
        to_json(bests, file_basename + "_bests.json")

        if plot:
            plt.savefig(file_basename + ".png", bbox_inches="tight")

    return model_logs, bests


def get_model_type(model_name: str, types: list = ["link", "split"]) -> str:
    for type in types:
        if type in model_name.lower():
            return type


def get_geo_mean(data, tags, smoothing=None):
    if smoothing is not None:
        for tag in tags:
            data[tag] = smooth(data[tag])
    temp_prod = np.ones_like(data[tags[0]])
    for tag in tags:
        temp_prod *= data[tag]
    return temp_prod ** (1 / len(tags))


def smooth(scalars, weight=0.99):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return np.array(smoothed)


def plot_geo_mean(model_logs):
    for model_name in model_logs.keys():
        plt.plot(
            model_logs[model_name]["step"],
            model_logs[model_name]["smooth_geo_mean"],
            label=model_name,
        )
    plt.legend()


def plot_scores(model_logs, tag="scores"):
    for model_name in model_logs.keys():
        plt.plot(
            model_logs[model_name]["score_steps"],
            model_logs[model_name][tag],
            label=f"{model_name} - best: {model_logs[model_name]['score_steps'][model_logs[model_name][tag].argmin()]}",
        )
    plt.legend()


def plot_all(model_logs, tags, size=7):
    plt.figure(figsize=(size, size * len(tags)))
    for i, tag in enumerate(tags):
        plt.subplot(len(tags), 1, i + 1, title=tag)
        plot_scores(model_logs, tag)


def show_best_steps(model_logs, types: list = ["link", "split"]):
    bests = defaultdict(dict)
    for model_name in model_logs.keys():
        this_best_score = model_logs[model_name]["scores"][
            model_logs[model_name]["score_steps"] == model_logs[model_name]["best_step"]
        ][0]
        print(
            f'{model_name} \n\t best step: {model_logs[model_name]["best_step"]} \n\t with score {this_best_score}'
        )

        type = get_model_type(model_name)
        if type not in bests.keys() or bests[type]["score"] > this_best_score:
            bests[type] = {
                "score": this_best_score,
                "model_name": model_name,
                "step": model_logs[model_name]["best_step"],
                "layer_name": get_best_layer(
                    model_name, model_logs[model_name]["best_step"]
                ),
            }

    for type in types:
        print(
            f'Best {type}: \n\t model_name: {bests[type]["model_name"]} \n\t layer_name: {bests[type]["layer_name"]} \n\t score: {bests[type]["score"]}'
        )

    return bests


def get_best_layer(model_name, step):
    return os.path.join(*model_name.split("_"), f"models/models_checkpoint_{step}")


# %%
if __name__ == "__main__":
    config_path = sys.argv[1]
    config = read_config(config_path)
    logs, bests = pick_checkpoints(**config)
