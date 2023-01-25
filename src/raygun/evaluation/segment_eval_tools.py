#%%
from collections import defaultdict
from glob import glob
import json
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import matplotlib

# switch to svg backend
matplotlib.use("svg")
# update latex preamble
plt.rcParams.update(
    {
        "svg.fonttype": "path",
        "font.family": "sans-serif",
        "font.sans-serif": "AvenirNextLTPro",  # ["Avenir", "AvenirNextLTPro", "Avenir Next LT Pro", "AvenirNextLTPro-Regular", 'UniversLTStd-Light', 'Verdana', 'Helvetica']
        "path.simplify": True,
        # "text.usetex": True,
        # "pgf.rcfonts": False,
        # "pgf.texsystem": 'pdflatex', # default is xetex
        # "pgf.preamble": [
        #      r"\usepackage[T1]{fontenc}",
        #      r"\usepackage{mathpazo}"
        #      ]
    }
)


def get_metric_dict(path="."):
    train_list = glob(os.path.join(path, f"train_*"))
    # predict_list = []
    metrics = {}
    metric_dict = {}  # metric_dict['train_ds', 'predict_ds'] = metric
    for folder in train_list:
        train_name = folder.split("/")[-1]
        with open(f"{folder}/metrics/metrics.json", "r") as f:
            metrics[folder] = json.load(f)
        for key, stats in metrics[folder].items():
            if not key.isnumeric():
                predict_name = key.replace("segment_", "").replace(train_name + "_", "")
                # predict_list.append(predict_name)
                if not "voi_split" in stats.keys():
                    stats = list(stats.values())[0]
                metric_dict[train_name, predict_name] = stats
    return metric_dict

#%%
def compare_metrics(
    metric_dict=None,
    norm_base=["train_real_30nm", "predict_real_90nm"],
    metrics=[
        "rand_split",
        "rand_merge",
        "voi_split",
        "voi_merge",
        "nvi_split",
        "nvi_merge",
    ],
):
    if metric_dict is None:
        metric_dict = get_metric_dict()
    elif isinstance(metric_dict, str):
        metric_dict = get_metric_dict(metric_dict)

    normed_metrics = defaultdict(dict)
    norm = metric_dict[tuple(norm_base)]
    for (train, predict), stats in metric_dict.items():
        if metrics[0] not in stats.keys():
            best_eval = {}
            for thresh, these_stats in stats.items():
                if len(best_eval) == 0 or get_score(best_eval) > get_score(these_stats):
                    best_eval = these_stats.copy()
                    best_eval["segment_ds"] = thresh
            stats = best_eval
        for metric in metrics:
            # print(f'Baseline for {metric} is {norm[metric]} from {norm_base}')
            normed_metrics[metric][train, predict] = stats[metric]  # / norm[metric]

    # plot_metrics(normed_metrics)
    return normed_metrics


def get_train_predict(keys):
    trains = []
    predicts = []
    for train, predict in keys:
        trains.append(train)
        predicts.append(predict)

    return set(trains), set(predicts), [predicts.count(pred) for pred in set(predicts)]


def plot_metrics(all_metrics):
    width = 0.35  # the width of the bars
    pad = 0.2

    color_dict = get_color_dict(all_metrics)

    fig, axs = plt.subplots(
        len(all_metrics.keys()), 1, figsize=(10, 10 * len(all_metrics.keys()))
    )
    for a, (met, metrics) in enumerate(all_metrics.items()):
        axs[a].set_xlabel(met)
        keys = metrics.keys()
        trains, predicts, predict_counts = get_train_predict(keys)
        # Train as bar color
        x = 0
        xs = []
        legend = set()
        for predict, count in zip(predicts, predict_counts):
            x += pad
            x += (count * width) / 2
            xs.append(x)
            i = 0 - (count // 2)
            for train in trains:
                if (train, predict) in keys:
                    legend.add(train)
                    this_x = x + (i * width)
                    axs[a].barh(
                        this_x,
                        metrics[train, predict],
                        width,
                        label=train,
                        # color=color_dict[train]
                    )
                    i += 1
            x += i * width
        # Predict as axis tick
        axs[a].set_yticks(xs)
        axs[a].set_yticklabels(predicts)
        # axs[a].legend(legend)
        axs[a].legend()
    plt.show()


def get_thresh_line(thresh_metrics, met, base="volumes/segmentation_"):
    split = []
    merge = []
    # ds_list = []
    threshes = sorted([ds.replace(base, "") for ds in thresh_metrics.keys()])
    for thresh in threshes:
        split.append(thresh_metrics[base + thresh][met + "_split"])
        merge.append(thresh_metrics[base + thresh][met + "_merge"])
    #     ds_list.append(base+thresh)
    # print(ds_list)
    return split, merge


def plot_metric_pairs_scatters(all_metrics, thresh_metrics=None, bests=[], mets=None):
    if mets is None:
        mets = set()
        for met in all_metrics.keys():
            mets.add(met.split("_")[0])

    # colors = list(TABLEAU_COLORS.values())
    # color_dict = get_color_dict(all_metrics)

    fig, axs = plt.subplots(len(mets), 1, figsize=(10, 10 * len(mets)))
    try:
        len(axs)
    except:
        axs = [axs]
    for a, met in enumerate(mets):
        axs[a].set_xlabel(met)
        keys = set(all_metrics[f"{met}_split"].keys())
        for key in set(all_metrics[f"{met}_merge"].keys()):
            keys.add(key)
        lim = 0
        if thresh_metrics is not None:
            for train, predict in keys:
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
                try:
                    this_thresh_metric = thresh_metrics[train, predict]
                except:
                    this_thresh_metric = thresh_metrics[train, predict + "."]
                split, merge = get_thresh_line(this_thresh_metric, met)
                if (train, predict) in bests or len(bests) == 0:
                    kwargs = {"linewidth": 1}
                else:
                    kwargs = {"linestyle": "dashed", "linewidth": 0.5, "alpha": 0.5}
                axs[a].plot(split, merge, color=color, **kwargs)
                lim = max([max(split), max(merge), lim])

        for train, predict in keys:
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

            split = all_metrics[f"{met}_split"][train, predict]
            merge = all_metrics[f"{met}_merge"][train, predict]
            lim = max([split, merge, lim])
            if (train, predict) in bests or len(bests) == 0:
                kwargs = {"color": color, "s": 95}
                _, _, train_ac = get_category(train)
                _, _, predict_ac = get_category(predict)
                # label = f'train on {train_ac} > predict on {predict_ac} (best)'
                label = f"{train_ac}-train | {predict_ac}-predict"
            else:
                # label = f'train on {train_ac} > predict on {predict_ac}'
                label = None
                kwargs = {"facecolors": "none", "edgecolors": color, "s": 70}
            axs[a].scatter(split, merge, label=label, marker=marker, **kwargs)
        axs[a].set_xlabel("Split")
        axs[a].set_ylabel("Merge")
        axs[a].set_title(met)
        axs[a].set_xlim([0, lim])
        axs[a].set_ylim([0, lim])
        # axs[a].legend(legend)
        axs[a].legend()  # bbox_to_anchor=(2, 1))
    plt.show()
    return fig


def plot_metric_pairs_bar(all_metrics):
    width = 0.35  # the width of the bars
    pad = 0.2
    mets = set()
    for met in all_metrics.keys():
        mets.add(met.split("_")[0])

    colors = list(TABLEAU_COLORS.values())
    color_dict = get_color_dict(all_metrics)

    fig, axs = plt.subplots(len(mets), 1, figsize=(10, 10 * len(mets)))
    for a, met in enumerate(mets):
        axs[a].set_xlabel(met)
        keys = set(all_metrics[f"{met}_split"].keys())
        for key in set(all_metrics[f"{met}_merge"].keys()):
            keys.add(key)
        trains, predicts, predict_counts = get_train_predict(keys)
        # Train as bar color
        x = 0
        xs = []
        legend = set()
        for predict, count in zip(predicts, predict_counts):
            x += pad
            x += (count * width) / 2
            xs.append(x)
            i = 0 - (count // 2)
            for train in trains:
                if (train, predict) in keys:
                    split = all_metrics[f"{met}_split"][train, predict]
                    merge = all_metrics[f"{met}_merge"][train, predict]
                    legend.add(train)
                    this_x = x + (i * width)
                    # color=color_dict[train]
                    if "split" in train:
                        color = "blue"  # colors[0]
                    elif "link" in train:
                        color = "red"  # colors[1]
                    elif "split" in predict:
                        color = "green"  # colors[2]
                    elif "link" in predict:
                        color = "orange"  # colors[3]
                    elif "90nm" in train and "90nm" in predict:
                        color = "magenta"
                    elif "30nm" in train and "90nm" in predict:
                        color = "brown"
                    elif "30nm" in train and "30nm" in predict:
                        color = "black"
                    axs[a].barh(
                        this_x,
                        split,
                        width,
                        label=train,
                        # color=color_dict[train]
                    )
                    axs[a].barh(
                        this_x,
                        merge,
                        width,
                        label=train,
                        # color=color_dict[train],
                        bottom=split,
                    )
                    i += 1
            x += i * width
        # Predict as axis tick
        # axs[a].set_yticks(xs)
        # axs[a].set_yticklabels(predicts)
        axs[a].set_xlabel("Split")
        axs[a].set_ylabel("Merge")
        axs[a].set_title(met)
        axs[a].set_xlim([0, 4])
        axs[a].set_ylim([0, 4])
        # axs[a].legend(legend)
        axs[a].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    plt.show()


def get_color_dict(all):
    trains = None
    for these in all.values():
        these_trains, _, _ = get_train_predict(these.keys())
        if trains is None:
            trains = these_trains
        else:
            for train in these_trains:
                trains.add(train)
    color_dict = {}
    colors = list(TABLEAU_COLORS.values())
    for i, train in enumerate(trains):
        color_dict[train] = colors[i]
    return color_dict


def get_category(name):
    category = ""
    acronym = ""
    if "split" in name or "link" in name:
        category += "fake "
        acronym += "f"
    else:
        category += "real "
        acronym += "r"

    if "90nm" in name:
        category += "low quality"
        acronym += "LQ"
    else:
        category += "high quality"
        acronym += "HQ"
    return f"{category} ({acronym})", category, acronym


def get_result_table(metric_dict, met="voi", best_suf="_sum", best_f=np.min):
    keys = set()
    names = defaultdict(list)
    scores = defaultdict(list)
    for (train_name, predict_name), metrics in metric_dict.items():
        if met + "_split" not in metrics.keys():
            best_eval = {}
            for thresh, these_metrics in metrics.items():
                if len(best_eval) == 0 or get_score(best_eval) > get_score(
                    these_metrics
                ):
                    best_eval = these_metrics.copy()
                    best_eval["segment_ds"] = thresh
            metrics = best_eval
        # print(train, predict)
        train_cat, _, _ = get_category(train_name)
        predict_cat, _, _ = get_category(predict_name)
        if "split" in train_name or "split" in predict_name:
            type = "Split"
        elif "link" in train_name or "link" in predict_name:
            type = "Linked (original)"
        elif "real_30nm" in train_name and "real_30nm" in predict_name:
            type = "Ideal"
        elif "real_90nm" in train_name and "real_90nm" in predict_name:
            type = "Paired"
        else:
            type = "Naive"

        keys.add((type, train_cat, predict_cat))
        names[type, train_cat, predict_cat].append((train_name, predict_name))
        scores[type, train_cat, predict_cat, met + "_split"].append(
            metrics[met + "_split"]
        )
        scores[type, train_cat, predict_cat, met + "_merge"].append(
            metrics[met + "_merge"]
        )
        scores[type, train_cat, predict_cat, met + "_sum"].append(
            metrics[met + "_split"] + metrics[met + "_merge"]
        )

    bests = []
    out_str = f"Type, Trained On, Predicted On, {met.upper()} - Split, {met.upper()} - Merge, {met.upper()} - Sum \n"
    for type, train_cat, predict_cat in keys:
        out_str += f"{type}, {train_cat}, {predict_cat}, "
        best_score = best_f(scores[type, train_cat, predict_cat, met + best_suf])
        best_ind = scores[type, train_cat, predict_cat, met + best_suf].index(
            best_score
        )
        bests.append(names[type, train_cat, predict_cat][best_ind])
        for suf in ["_split", "_merge", "_sum"]:
            these = scores[type, train_cat, predict_cat, met + suf]
            mean = np.mean(these)
            best = these[best_ind]
            scores[type, train_cat, predict_cat, met + suf, "mean"] = mean
            scores[type, train_cat, predict_cat, met + suf, "best"] = best
            if len(these) > 1:
                out_str += f"%.3f (mean = %.3f), " % (best, mean)
            else:
                out_str += f"%.3f, " % (best)
        out_str = out_str[:-2] + "\n"
    print(out_str)
    return scores, bests


# %%
# if __name__=='__main__':
os.chdir("/n/groups/htem/Segmentation/networks/xray_setups/eccv-bic/setup02_2")

metric_dict = get_metric_dict()

include = [
    "link20220407s4c310000",
    "real",
    "real",
    "split20220407s42c340000",
]

# keys = list(metric_dict.keys())
# for train, predict in keys:
#     if train.replace('train_', '')[:-5] not in include or predict.replace('predict_', '')[:-5] not in include:
#         del metric_dict[train, predict]
# all_metrics = compare_metrics(metric_dict)

sys.path.append(os.getcwd)
from batch_evaluate import *

thresh_metrics = batch_evaluate()
keys = list(thresh_metrics.keys())
for train, predict in keys:
    if (
        train.replace("train_", "")[:-5] not in include
        or predict.replace("predict_", "")[:-5] not in include
    ):
        del thresh_metrics[train, predict]
all_metrics = compare_metrics(thresh_metrics)


#%%
# scores, bests = get_result_table(metric_dict, met='voi', best_suf='_sum', best_f=np.min)
scores, bests = get_result_table(
    thresh_metrics, met="voi", best_suf="_sum", best_f=np.min
)

#%%
# bests = [('train_real_30nm', 'predict_real_30nm'),
#         ('train_real_90nm', 'predict_real_90nm'),
#         ('train_link20220407s42c310000_90nm', 'predict_real_90nm'),
#         ('train_split20220407s42c340000_90nm', 'predict_real_90nm'),
#         ('train_real_30nm', 'predict_link20220407s42c310000_30nm'),
#         ('train_real_30nm', 'predict_split20220407s42c340000_30nm')
#         ]
plot_metric_pairs_scatters(all_metrics, thresh_metrics=thresh_metrics, bests=bests)
# %%
fig = plot_metric_pairs_scatters(
    all_metrics, thresh_metrics=thresh_metrics, bests=bests, mets=["voi"]
)
