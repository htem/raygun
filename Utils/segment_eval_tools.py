#%%
from glob import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def get_metric_dict(path='.'):
    train_list = glob(os.path.join(path, f'train_*'))
    predict_list = []
    metrics = {}
    metric_dict = {} # metric_dict['train_ds', 'predict_ds'] = metric
    for folder in train_list:
        train_name = folder.split('/')[-1]
        with open(f'{folder}/metrics/metrics.json', 'r') as f:
            metrics[folder] = json.load(f)
        for key, stats in metrics[folder].items():
            if not key.isnumeric():
                predict_name = key.replace('segment_', '').replace(train_name+'_', '')
                predict_list.append(predict_name)
                metric_dict[train_name, predict_name] = stats
    return metric_dict

def compare_metrics(metric_dict, 
                    norm_base=['train_real_30nm', 'predict_real_90nm'],
                    metrics=["rand_split", "rand_merge", "voi_split", "voi_merge", "nvi_split", "nvi_merge"]):
    
    normed_metrics = {}
    for metric in metrics:
        norm = metric_dict[tuple(norm_base)]
        for (train, predict), stats in metric_dict.items():
            normed_metrics[metric][train, predict] = stats[metric] / norm # TODO: Decide how to adjust with baseline

    plot_metrics(normed_metrics)
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
    pad = 0.1
    
    fig, axs = plt.subplots(1, len(all_metrics.keys()), sharex=True, figsize=(10*len(all_metrics.keys()), 10))
    for met, metrics in all_metrics.items():
        axs.set_xlabel(met)
        keys = metrics.keys()
        trains, predicts, predict_counts = get_train_predict(keys)
        #Predict as axis tick
        #Train as bar color        
        x = pad
        for predict, count in zip(predicts, predict_counts):
            x += ...
        

# %%
