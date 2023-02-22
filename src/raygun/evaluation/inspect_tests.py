#%%
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from raygun import read_config
from glob import glob
import os
import sys
from scipy.stats import (
    variation,
    kstest,
    ks_2samp,
    anderson,
    wilcoxon,
    mannwhitneyu,
    epps_singleton_2samp,
)

from raygun.utils import load_json_file, to_json

# # switch to svg backend
# matplotlib.use("svg")
# # update latex preamble
# plt.rcParams.update(
#     {
#         "svg.fonttype": "path",
#         "font.family": "sans-serif",
#         # "font.sans-serif": "AvenirNextLTPro",  # ["Avenir", "AvenirNextLTPro", "Avenir Next LT Pro", "AvenirNextLTPro-Regular", 'UniversLTStd-Light', 'Verdana', 'Helvetica']
#         "path.simplify": True,
#         # "text.usetex": True,
#         # "pgf.rcfonts": False,
#         # "pgf.texsystem": 'pdflatex', # default is xetex
#         # "pgf.preamble": [
#         #      r"\usepackage[T1]{fontenc}",
#         #      r"\usepackage{mathpazo}"
#         #      ]
#     }
# )


def test_stats(raw_dict, stat=kstest):
    stats = {}
    for metric, results in raw_dict.items():
        stats[metric] = {"keys": list(results.keys())}
        stats[metric]["p_merge"] = np.ones(
            (len(stats[metric]["keys"]), len(stats[metric]["keys"]))
        )
        stats[metric]["p_split"] = np.ones(
            (len(stats[metric]["keys"]), len(stats[metric]["keys"]))
        )
        for i in range(len(stats[metric]["keys"])):
            # for j in range(i + 1, len(stats[metric]["keys"])):
            for j in range(i, len(stats[metric]["keys"])):
                result = np.array(results[stats[metric]["keys"][i]])
                result2 = np.array(results[stats[metric]["keys"][j]])
                stats[metric]["p_merge"][i, j] = stat(result[:, 1], result2[:, 1])[1]
                stats[metric]["p_split"][i, j] = stat(result[:, 0], result2[:, 0])[1]
                stats[metric]["p_merge"][j, i] = stats[metric]["p_merge"][i, j]
                stats[metric]["p_split"][j, i] = stats[metric]["p_split"][i, j]
    return stats


def plot_stats(stats, size=5):
    fig, axes = plt.subplots(
        len(stats.keys()), 2, figsize=(2 * size, len(stats.keys()) * size)
    )
    for ax, (metric, data) in zip(axes, stats.items()):
        p_merge = data["p_merge"]
        p_split = data["p_split"]
        keys = data["keys"]
        ax[0].imshow(p_merge, cmap="hot", interpolation="nearest")
        ax[0].set_xticks(np.arange(len(keys)))
        ax[0].set_yticks(np.arange(len(keys)))
        ax[0].set_xticklabels(["\n".join(k) for k in keys], rotation=90)
        ax[0].set_yticklabels(["\n".join(k) for k in keys])
        ax[0].set_title(f"{metric} merge")
        ax[0].set_xlabel("Train on:\nPredict on:")
        ax[0].set_ylabel("Train on:\nPredict on:")
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax[0].text(
                    j, i, f"{p_merge[i, j]:.2f}", ha="center", va="center", color="w"
                )
        ax[1].imshow(p_split, cmap="hot", interpolation="nearest")
        ax[1].set_xticks(np.arange(len(keys)))
        ax[1].set_yticks(np.arange(len(keys)))
        ax[1].set_xticklabels(["\n".join(k) for k in keys], rotation=90)
        ax[1].set_yticklabels(["\n".join(k) for k in keys])
        ax[1].set_title(f"{metric} split")
        ax[1].set_xlabel("Train on:\nPredict on:")
        ax[1].set_ylabel("Train on:\nPredict on:")
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax[1].text(
                    j, i, f"{p_split[i, j]:.2f}", ha="center", va="center", color="w"
                )
    fig.tight_layout()
    fig.show()
    return fig


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
    meta_test_dir="/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN/tensorboards",
    tags=None,
    smoothing=0,
    plot=True,
    save=False,
    types: list = ["link", "split", "real_90nm", "real_30nm"],
):
    model_tests, file_basename, tags = load_json_tests(meta_test_dir, tags)

    for model_name in model_tests.keys():
        model_tests[model_name]["geo_mean"] = get_geo_mean(
            model_tests[model_name], tags
        )
        model_tests[model_name]["score"] = get_sum(
            model_tests[model_name], tags, smoothing=smoothing
        )

    bests = show_best_steps(model_tests, types)
    if plot:
        plot_all(model_tests, tags + ["score"])

    if save:
        to_json(model_tests, file_basename + ".json")
        to_json(bests, file_basename + "_bests.json")

        if plot:
            plt.savefig(file_basename + ".svg", bbox_inches="tight")

    return model_tests, bests


def get_sum(data, tags, smoothing=None):
    if smoothing is not None and smoothing > 0:
        for tag in tags:
            data[tag] = smooth(data[tag], smoothing)
    this_sum = np.zeros_like(data[tags[0]])
    for tag in tags:
        this_sum += data[tag]
    return this_sum


def get_model_type(
    model_name: str, types: list = ["link", "split", "real_90nm", "real_30nm"]
) -> str:
    for type in types:
        if type in model_name.lower():
            return type


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


def get_type(name):
    if "link" in name:
        return "link"
    elif "split" in name:
        return "split"
    elif "real_" in name:
        return (
            name.replace("real_", "real").replace("train_", "").replace("predict_", "")
        )
    else:
        return name.replace("train_", "").replace("predict_", "")


def make_data_dict(tests):
    raw_dict = defaultdict(lambda: defaultdict(list))
    for name, results in tests.items():
        train, predict = name.split("_predict_")
        train = get_type(train)
        predict = get_type(predict)
        metrics = [key for key in results.keys() if "_split" in key]
        for metric in metrics:
            raw_dict[metric.replace("_split", "")][train, predict].append(
                [results[metric], results[metric.replace("_split", "_merge")]]
            )

    sums = defaultdict(defaultdict)
    means = defaultdict(defaultdict)
    mins = defaultdict(defaultdict)
    maxs = defaultdict(defaultdict)
    for metric, results in raw_dict.items():
        for (train, predict), data in results.items():
            sums[metric][train, predict] = np.sum(data, 1)
            means[metric][train, predict] = np.mean(data, 0)
            mins[metric][train, predict] = np.min(data, 0)
            maxs[metric][train, predict] = np.max(data, 0)

    return (
        raw_dict,
        sums,
        means,
        mins,
        maxs,
    )  # raw_dict[train_type, predict_type][metric] = [split, merge]


def plot_metric_pairs_scatters(data):
    metrics = list(data.keys())

    # colors = list(TABLEAU_COLORS.values())
    # color_dict = get_color_dict(all_metrics)

    fig, axs = plt.subplots(len(data), 1, figsize=(10, 10 * len(data)))
    try:
        len(axs)
    except:
        axs = [axs]
    for a, (met, results) in enumerate(data.items()):
        for (train, predict), result in results.items():
            # color=color_dict[train]
            if "split" in train:
                marker = "v"
                color = "blue"  # colors[0]
            elif "link" in train:
                marker = "v"
                color = "red"  # colors[1]
            elif "split" in predict:
                marker = "^"
                color = "green"  # colors[2]
            elif "link" in predict:
                marker = "^"
                color = "orange"  # colors[3]
            elif "90nm" in train and "90nm" in predict:
                marker = "o"
                color = "magenta"
            elif "30nm" in train and "90nm" in predict:
                marker = "X"
                color = "brown"
            elif "30nm" in train and "30nm" in predict:
                marker = "D"
                color = "black"

            # lim = max(result + [0])
            # if (train, predict) in bests or len(bests) == 0:
            kwargs = {"color": color, "s": 95}
            label = f"{train}-train | {predict}-predict"
            #     _, _, train_ac = get_category(train)
            #     _, _, predict_ac = get_category(predict)
            #     # label = f'train on {train_ac} > predict on {predict_ac} (best)'
            #     label = f"{train_ac}-train | {predict_ac}-predict"
            # else:
            #     # label = f'train on {train_ac} > predict on {predict_ac}'
            #     label = None
            #     kwargs = {"facecolors": "none", "edgecolors": color, "s": 70}
            result = np.array(result)
            if len(result.shape) == 1:
                axs[a].scatter(
                    result[0], result[1], label=label, marker=marker, **kwargs
                )
            else:
                axs[a].scatter(
                    result[:, 0], result[:, 1], label=label, marker=marker, **kwargs
                )

        axs[a].set_xlabel("Split")
        axs[a].set_ylabel("Merge")
        axs[a].set_title(met)
        # axs[a].set_xlim([0, lim])
        # axs[a].set_ylim([0, lim])
        # axs[a].legend(legend)
        axs[a].legend()  # bbox_to_anchor=(2, 1))
    plt.show()
    return fig


# %%
if __name__ == "__main__":
    config_path = sys.argv[1]
    config = read_config(config_path)
    tests, bests = show_data(**config)
