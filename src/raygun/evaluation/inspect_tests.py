#%%
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from raygun import read_config
from glob import glob
import os
import sys

from raygun.utils import load_json_file, to_json


def convert_json_test(path: str, tags=None):
    old = load_json_file(path)
    metrics = {}
    step = np.array([k for k in old.keys()])[0]
    if tags is None:
        tags = [k for k in old[step].keys()]

    for tag in tags:
        val = old[step][tag]
        if type(val) is not dict:  # TODO: hacky
            metrics[tag] = val
        else:
            tags.remove(tag)

    return metrics, tags


#%%


def load_json_tests(paths, tags=None):
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
    model_tests = {}  # model_name: test_metrics
    for file in files:
        model_name = "_".join(
            file.replace(base_path, "")
            .lstrip("/")
            .rstrip(".json")
            .rstrip(base_name)
            .rstrip("/")
            .split("/")
        )
        model_tests[model_name], tags = convert_json_test(file, tags)

    return model_tests, file_basename, tags


def show_data(
    meta_test_dir="/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN_7/tensorboards",
    tags=None,
    smoothing=0.999,
    plot=True,
    save=False,
    types: list = ["link", "split", "real_90nm", "real_30nm"],
):
    model_tests, file_basename, tags = load_json_tests(meta_test_dir, tags)

    for model_name in model_tests.keys():
        model_tests[model_name]["geo_mean"] = get_geo_mean(
            model_tests[model_name], tags
        )
        # model_tests[model_name]['smooth_geo_mean'] = smooth(model_tests[model_name]['geo_mean'], smoothing)
        model_tests[model_name]["scores"] = get_geo_mean(
            model_tests[model_name], tags, smoothing=smoothing
        )

        model_tests[model_name]["best_step"] = model_tests[model_name]["score_steps"][
            model_tests[model_name]["scores"].argmin()
        ]
        for tag in tags + ["geo_mean"]:
            model_tests[model_name][tag] = model_tests[model_name][tag][inds]

    bests = show_best_steps(model_tests, types)
    if plot:
        plot_all(model_tests, tags + ["scores"])

    if save:
        to_json(model_tests, file_basename + ".json")
        to_json(bests, file_basename + "_bests.json")

        if plot:
            plt.savefig(file_basename + ".png", bbox_inches="tight")

    return model_tests, bests


def get_model_type(
    model_name: str, types: list = ["link", "split", "real_90nm", "real_30nm"]
) -> str:
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


def plot_geo_mean(model_tests):
    for model_name in model_tests.keys():
        plt.plot(
            model_tests[model_name]["step"],
            model_tests[model_name]["smooth_geo_mean"],
            label=model_name,
        )
    plt.legend()


def plot_scores(model_tests, tag="scores"):
    for model_name in model_tests.keys():
        plt.plot(
            model_tests[model_name]["score_steps"],
            model_tests[model_name][tag],
            label=f"{model_name} - best: {model_tests[model_name]['score_steps'][model_tests[model_name][tag].argmin()]}",
        )
    plt.legend()


def plot_all(model_tests, tags, size=7):
    plt.figure(figsize=(size, size * len(tags)))
    for i, tag in enumerate(tags):
        plt.subplot(len(tags), 1, i + 1, title=tag)
        plot_scores(model_tests, tag)


def show_best_steps(
    model_tests, types: list = ["link", "split", "real_90nm", "real_30nm"]
):
    bests = defaultdict(dict)
    for model_name in model_tests.keys():
        this_best_score = model_tests[model_name]["scores"][
            model_tests[model_name]["score_steps"]
            == model_tests[model_name]["best_step"]
        ][0]
        print(
            f'{model_name} \n\t best step: {model_tests[model_name]["best_step"]} \n\t with score {this_best_score}'
        )

        type = get_model_type(model_name, types)
        if type not in bests.keys() or bests[type]["score"] > this_best_score:
            bests[type] = {
                "score": this_best_score,
                "model_name": model_name,
                "step": model_tests[model_name]["best_step"],
                "layer_name": get_best_layer(
                    model_name, model_tests[model_name]["best_step"]
                ),
            }

    for type in bests.keys():
        print(
            f'Best {type}: \n\t model_name: {bests[type]["model_name"]} \n\t layer_name: {bests[type]["layer_name"]} \n\t score: {bests[type]["score"]}'
        )

    return bests


def get_best_layer(model_name, step):
    return os.path.join(*model_name.split("_"), f"models/models_checkpoint_{step}")


def inspect_tests(config_path=None):
    if config_path is None:
        config_path = sys.argv[1]
    config = read_config(config_path)
    return show_data(**config)


#%%
means = defaultdict(lambda: defaultdict(defaultdict))
maxs = defaultdict(lambda: defaultdict(defaultdict))
mins = defaultdict(lambda: defaultdict(defaultdict))
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for r, res in enumerate(["split", "merge"]):
    for c, metric in enumerate([tag for tag in tags if res in tag]):
        axes[r, c].set_title(" ".join(metric.split("_")))
        if c == 0:
            axes[r, c].set_ylabel(res)
        for x, (name, metrics) in enumerate(model_tests.items()):
            vals = [v[metric] for v in metrics[res].values()]
            means[metric][res][name] = np.mean(vals)
            maxs[metric][res][name] = np.max(vals)
            mins[metric][res][name] = np.min(vals)
            axes[r, c].scatter(
                np.ones_like(vals) * x,
                vals,
                label=f"{name} (mean={means[metric][res][name]:10.4f}, min={mins[metric][res][name]:10.4f}, max={maxs[metric][res][name]:10.4f}",
            )

        axes[r, c].set_xticklabels(
            [
                f"{n}\nmean={means[metric][res][n]:3.4f}\nmin={mins[metric][res][n]:3.4f}\nmax={maxs[metric][res][n]:3.4f}"
                for n in model_tests.keys()
            ]
        )
fig.tight_layout()

# %%
if __name__ == "__main__":
    config_path = sys.argv[1]
    config = read_config(config_path)
    tests, bests = show_data(**config)
