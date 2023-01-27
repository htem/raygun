#%%
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from raygun import read_config
from glob import glob
import os
import sys

from raygun.utils import load_json_file, to_json


def convert_json_log(path: str, tags=None):
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


def load_json_logs(paths, tags=None):
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
    model_logs = {}  # model_name: log_metrics
    for file in files:
        model_name = "_".join(
            file.replace(base_path, "")
            .lstrip("/")
            .rstrip(".json")
            .rstrip(base_name)
            .rstrip("/")
            .split("/")
        )
        model_logs[model_name], tags = convert_json_log(file, tags)

    return model_logs, file_basename, tags


def load_tensorboards(  # TODO: Cleanup
    meta_log_dir="/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN/tensorboards",
    start=2000,  # TODO: ALLOW FOR UNBOUNDED
    tags=None,
):
    if tags is None:
        tags = [
            "l1_loss/cycled_A",
            "l1_loss/cycled_B",
            "gan_loss/fake_A",
            "gan_loss/fake_B",
        ]
    meta_log_dir = meta_log_dir.rstrip("/*") + "/*"

    folders = glob(meta_log_dir)
    file_basename = os.path.join(
        os.path.dirname(os.path.commonpath(folders)), "model_logs"
    )
    model_logs = {}  # model_name: log_metrics
    for folder in folders:
        log_paths = glob(folder + "/*")
        log_path = max(log_paths, key=os.path.getctime)
        model_name = "_".join(
            folder.replace(os.path.commonpath(folders), "").lstrip("/").split("/")
        )
        model_logs[model_name] = parse_events_file(log_path, tags)
        # check what we want is there:
        p = 0
        while start not in model_logs[model_name]["step"] and p < len(log_paths):
            model_logs[model_name] = parse_events_file(log_paths[p], tags)
            p += 1

    return model_logs, file_basename, tags


def pick_checkpoints(
    meta_log_dir="/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN/tensorboards",
    increment=2000,
    start=2000,  # TODO: ALLOW FOR UNBOUNDED
    final=200000,
    tags=None,
    smoothing=0.999,
    plot=True,
    save=False,
    tensorboard=True,
    types: list = ["link", "split", "real_90nm", "real_30nm"],
):
    if tensorboard:  # TODO: Make cleaner, this is super hacky
        model_logs, file_basename, tags = load_tensorboards(meta_log_dir, start, tags)
    else:
        model_logs, file_basename, tags = load_json_logs(meta_log_dir, tags)

    for model_name in model_logs.keys():
        model_logs[model_name]["geo_mean"] = get_geo_mean(model_logs[model_name], tags)
        # model_logs[model_name]['smooth_geo_mean'] = smooth(model_logs[model_name]['geo_mean'], smoothing)
        # model_logs[model_name]["smooth_geo_mean"] = get_geo_mean(
        #     model_logs[model_name], tags, smoothing=smoothing
        # )
        model_logs[model_name]["smooth_sum"] = get_sum(
            model_logs[model_name], tags, smoothing=smoothing
        )

        inds = np.array(
            [
                np.argmax(model_logs[model_name]["step"] == step)
                for step in np.arange(start, final + increment, increment)
                if step in model_logs[model_name]["step"]
            ]
        ).flatten()

        model_logs[model_name]["score_steps"] = np.arange(
            start, final + increment, increment
        )[: len(inds)]
        # model_logs[model_name]["scores"] = model_logs[model_name]["smooth_geo_mean"][
        #     inds
        # ]
        model_logs[model_name]["scores"] = model_logs[model_name]["smooth_sum"][inds]
        model_logs[model_name]["best_step"] = model_logs[model_name]["score_steps"][
            model_logs[model_name]["scores"].argmin()
        ]
        for tag in tags + ["geo_mean"]:
            model_logs[model_name][tag] = model_logs[model_name][tag][inds]

    bests = show_best_steps(model_logs, types)
    if plot:
        plot_all(model_logs, tags + ["scores"])

    if save:
        to_json(model_logs, file_basename + ".json")
        to_json(bests, file_basename + "_bests.json")

        if plot:
            plt.savefig(file_basename + ".png", bbox_inches="tight")

    return model_logs, bests


def get_model_type(
    model_name: str, types: list = ["link", "split", "real_90nm", "real_30nm"]
) -> str:
    for type in types:
        if type in model_name.lower():
            return type


def get_sum(data, tags, smoothing=None):
    if smoothing is not None and smoothing > 0:
        for tag in tags:
            data[tag] = smooth(data[tag], smoothing)
    this_sum = np.zeros_like(data[tags[0]])
    for tag in tags:
        this_sum += data[tag]
    return this_sum


def get_geo_mean(data, tags, smoothing=None):
    if smoothing is not None and smoothing > 0:
        for tag in tags:
            data[tag] = smooth(data[tag], smoothing)
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


def show_best_steps(
    model_logs, types: list = ["link", "split", "real_90nm", "real_30nm"]
):
    bests = defaultdict(dict)
    for model_name in model_logs.keys():
        this_best_score = model_logs[model_name]["scores"][
            model_logs[model_name]["score_steps"] == model_logs[model_name]["best_step"]
        ][0]
        print(
            f'{model_name} \n\t best step: {model_logs[model_name]["best_step"]} \n\t with score {this_best_score}'
        )

        type = get_model_type(model_name, types)
        if type not in bests.keys() or bests[type]["score"] > this_best_score:
            bests[type] = {
                "score": this_best_score,
                "model_name": model_name,
                "step": model_logs[model_name]["best_step"],
                "layer_name": get_best_layer(
                    model_name, model_logs[model_name]["best_step"]
                ),
            }

    for type in bests.keys():
        print(
            f'Best {type}: \n\t model_name: {bests[type]["model_name"]} \n\t layer_name: {bests[type]["layer_name"]} \n\t score: {bests[type]["score"]}'
        )

    return bests


def get_best_layer(model_name, step):
    return os.path.join(*model_name.split("_"), f"models/models_checkpoint_{step}")


def inspect_logs(config_path=None):
    if config_path is None:
        config_path = sys.argv[1]
    config = read_config(config_path)
    return pick_checkpoints(**config)


# %%
if __name__ == "__main__":
    config_path = sys.argv[1]
    config = read_config(config_path)
    logs, bests = pick_checkpoints(**config)
