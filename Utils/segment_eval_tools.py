#%%
from collections import defaultdict
from glob import glob
import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np

def get_metric_dict(path='.'):
    train_list = glob(os.path.join(path, f'train_*'))
    # predict_list = []
    metrics = {}
    metric_dict = {} # metric_dict['train_ds', 'predict_ds'] = metric
    for folder in train_list:
        train_name = folder.split('/')[-1]
        with open(f'{folder}/metrics/metrics.json', 'r') as f:
            metrics[folder] = json.load(f)
        for key, stats in metrics[folder].items():
            if not key.isnumeric():
                predict_name = key.replace('segment_', '').replace(train_name+'_', '')
                # predict_list.append(predict_name)
                metric_dict[train_name, predict_name] = stats
    return metric_dict

def compare_metrics(metric_dict=None, 
                    norm_base=['train_real_30nm', 'predict_real_90nm'],
                    metrics=["rand_split", "rand_merge", "voi_split", "voi_merge", "nvi_split", "nvi_merge"]):
    if metric_dict is None:
        metric_dict = get_metric_dict()
    elif isinstance(metric_dict, str):
        metric_dict = get_metric_dict(metric_dict)

    normed_metrics = defaultdict(dict)
    norm = metric_dict[tuple(norm_base)]
    for metric in metrics:
        print(f'Baseline for {metric} is {norm[metric]} from {norm_base}')
        for (train, predict), stats in metric_dict.items():
            normed_metrics[metric][train, predict] = stats[metric] #/ norm[metric] # TODO: Decide how to adjust with baseline

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
    
    fig, axs = plt.subplots(len(all_metrics.keys()), 1, figsize=(10, 10*len(all_metrics.keys())))
    for a, (met, metrics) in enumerate(all_metrics.items()):
        axs[a].set_xlabel(met)
        keys = metrics.keys()
        trains, predicts, predict_counts = get_train_predict(keys)
        #Train as bar color        
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
                    this_x = x + (i * width) / 2
                    axs[a].barh(this_x, 
                                metrics[train, predict], 
                                width, 
                                label=train, 
                                color=color_dict[train])
                    i += 1
        #Predict as axis tick
        axs[a].set_yticks(xs)
        axs[a].set_yticklabels(predicts)
        axs[a].legend(legend)
        print(xs)
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
# %%
