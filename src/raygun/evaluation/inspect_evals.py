#%%
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from raygun import read_config
from glob import glob
import os
import sys

from raygun.utils import load_json_file, to_json


def convert_json_eval(path: str, tags=None):
    old = load_json_file(path)
    metrics = defaultdict(list)
    steps = [k for k in old.keys()]
    steps.sort()
    metrics["step"] = np.array(steps).astype(int)
    if tags is None:
        tags = [k for k in old[steps[0]].keys()]

    for step in steps:
        for tag in tags:
            val = old[step][tag]
            if type(val) is not dict:  # TODO: hacky
                metrics[tag].append(val)
            else:
                tags.remove(tag)

    for k, v in metrics.items():
        metrics[k] = np.array(v)

    return metrics, tags


#%%


def parse_events_file(path: str, tags: list):
    import tensorflow as tf  # put here to prevent loading tensorflow for every raygun operation

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


def load_json_evals(paths, tags=None):
    if isinstance(paths, str):
        paths = [paths]

    files = []
    for path in paths:
        files += glob(path)

    base_path = os.path.commonpath(files)
    base_name = os.path.commonprefix([file.split("/")[-1] for file in files]).replace(
        ".json", ""
    )
    file_basename = os.path.join(base_path, base_name)
    model_evals = {}  # model_name: eval_metrics
    for file in files:
        model_name = "_".join(
            file.replace(base_path, "")
            .lstrip("/")
            .rstrip(".json")
            .rstrip(base_name)
            .rstrip("/")
            .split("/")
        )
        model_evals[model_name], tags = convert_json_eval(file, tags)

    return model_evals, file_basename, tags


def pick_checkpoints(
    meta_eval_dir,
    tags=None,
    smoothing=0.999,
    plot=True,
    save=False,
    types: list = ["link", "split", "real_90nm", "real_30nm"],
):
    model_evals, file_basename, tags = load_json_evals(meta_eval_dir, tags)

    for model_name in model_evals.keys():
        model_evals[model_name]["geo_mean"] = get_geo_mean(
            model_evals[model_name], tags
        )
        model_evals[model_name]["smooth_geo_mean"] = get_geo_mean(
            model_evals[model_name], tags, smoothing=smoothing
        )
        model_evals[model_name]["smooth_sum"] = get_sum(
            model_evals[model_name], tags, smoothing=smoothing
        )
        # model_evals[model_name]["score"] = model_evals[model_name]["smooth_geo_mean"][
        #     inds
        # ]
        model_evals[model_name]["score"] = model_evals[model_name]["smooth_sum"]
        model_evals[model_name]["best_score"] = model_evals[model_name]["score"][
            model_evals[model_name]["score"].argmin()
        ]
        model_evals[model_name]["best_step"] = model_evals[model_name]["step"][
            model_evals[model_name]["score"].argmin()
        ]

    bests = show_best_steps(model_evals, types)
    if plot:
        plot_all(model_evals, tags + ["score"])

    if save:
        to_json(model_evals, file_basename + ".json")
        to_json(bests, file_basename + "_bests.json")

        if plot:
            plt.savefig(file_basename + ".png", bbox_inches="tight")

    return model_evals, bests


def get_model_type(
    model_name: str, types: list = ["link", "split", "real_90nm", "real_30nm"]
) -> str:
    for type in types:
        if type in model_name.lower():
            return type


def get_sum(data, tags, smoothing=None):
    if smoothing is not None:
        for tag in tags:
            data[tag] = smooth(data[tag])
    this_sum = np.zeros_like(data[tags[0]])
    for tag in tags:
        this_sum += data[tag]
    return this_sum


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


def plot_geo_mean(model_evals):
    for model_name in model_evals.keys():
        plt.plot(
            model_evals[model_name]["step"],
            model_evals[model_name]["smooth_geo_mean"],
            label=model_name,
        )
    plt.legend()


def plot_scores(model_evals, tag="scores"):
    for model_name in model_evals.keys():
        plt.plot(
            model_evals[model_name]["step"],
            model_evals[model_name][tag],
            label=f"{model_name} - best: {model_evals[model_name]['step'][model_evals[model_name][tag].argmin()]}",
        )
    plt.legend()


def plot_all(model_evals, tags, size=7):
    plt.figure(figsize=(size, size * len(tags)))
    for i, tag in enumerate(tags):
        plt.subplot(len(tags), 1, i + 1, title=tag)
        plot_scores(model_evals, tag)


def show_best_steps(
    model_evals, types: list = ["link", "split", "real_90nm", "real_30nm"]
):
    bests = defaultdict(dict)
    for model_name in model_evals.keys():
        this_best_score = model_evals[model_name]["best_score"]
        print(
            f'{model_name} \n\t best step: {model_evals[model_name]["best_step"]} \n\t with score {this_best_score}'
        )

        type = get_model_type(model_name, types)
        if type not in bests.keys() or bests[type]["score"] > this_best_score:
            bests[type] = {
                "score": this_best_score,
                "model_name": model_name,
                "step": model_evals[model_name]["best_step"],
                "layer_name": get_best_layer(
                    model_name, model_evals[model_name]["best_step"]
                ),
            }

    for type in bests.keys():
        print(
            f'Best {type}: \n\t model_name: {bests[type]["model_name"]} \n\t layer_name: {bests[type]["layer_name"]} \n\t score: {bests[type]["score"]}'
        )

    return bests


def get_best_layer(model_name, step):
    return os.path.join(*model_name.split("_"), f"models/models_checkpoint_{step}")


def inspect_evals(config_path=None):
    if config_path is None:
        config_path = sys.argv[1]
    config = read_config(config_path)
    return pick_checkpoints(**config)


# %%
if __name__ == "__main__":
    config_path = sys.argv[1]
    config = read_config(config_path)
    evals, bests = pick_checkpoints(**config)
