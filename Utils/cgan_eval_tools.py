#%%
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob
import os

def parse_events_file(path: str, tags: list):
    metrics = defaultdict(list)
    for e in tf.compat.v1.train.summary_iterator(path):
        for v in e.summary.value:
            if v.tag in tags:
                if v.tag == tags[0]:
                    metrics['step'].append(e.step)
                metrics[v.tag].append(v.simple_value)
    for k, v in metrics.items():
        metrics[k] = np.array(v)
    return metrics

# %%
def pick_checkpoints(meta_log_dir='/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/eccv-bic-2022/tensorboard/', 
                    increment=10000,
                    start=310000, #TODO: ALLOW FOR UNBOUNDED
                    final=350000, 
                    types=['link', 'split'], 
                    tags_dict={
                        'link': ['Cycle_Loss/A', 'Cycle_Loss/B'],
                        'split': ['l1_loss/cycled_A', 'l1_loss/cycled_A']
                    },
                    smoothing=0.999):
    if meta_log_dir[-2:] != '/*':
        if meta_log_dir[-1] != '/':
            meta_log_dir += '/*'
        else:
            meta_log_dir += '*'

    folders = glob(meta_log_dir)
    model_logs = {} #model_name: log_metrics
    for folder in folders:
        log_paths = glob(folder+'/*')
        log_path = max(log_paths, key=os.path.getctime)
        model_name = folder.split('/')[-1]
        model_type = get_model_type(model_name, types)
        model_logs[model_name] = parse_events_file(log_path, tags_dict[model_type])
        # check what we want is there:
        p = 0
        while start not in model_logs[model_name]['step'] and p < len(log_paths):
            model_logs[model_name] = parse_events_file(log_paths[p], tags_dict[model_type])
            p += 1

        model_logs[model_name]['geo_mean'] = get_geo_mean(model_logs[model_name], tags_dict[model_type])
        # model_logs[model_name]['smooth_geo_mean'] = smooth(model_logs[model_name]['geo_mean'], smoothing)
        model_logs[model_name]['smooth_geo_mean'] = get_geo_mean(model_logs[model_name], tags_dict[model_type], smoothing=smoothing)
    
    for model_name in model_logs.keys():
        inds = np.array([np.where(model_logs[model_name]['step'] == step) for step in np.arange(start, final+increment, increment)]).flatten()
        model_logs[model_name]['score_steps'] = np.arange(start, final+increment, increment)
        model_logs[model_name]['scores'] = model_logs[model_name]['smooth_geo_mean'][inds]
        model_logs[model_name]['best_step'] = model_logs[model_name]['score_steps'][model_logs[model_name]['scores'].argmin()]

    show_best_steps(model_logs)
    return model_logs

def get_model_type(model_name: str, types: list=['link', 'split']) -> str:
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
    return temp_prod**(1/len(tags))

def smooth(scalars, weight=0.99):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

def plot_geo_mean(model_logs):
    for model_name in model_logs.keys():
        plt.plot(model_logs[model_name]['step'], model_logs[model_name]['smooth_geo_mean'], label=model_name)
    plt.legend()

def plot_scores(model_logs):
    for model_name in model_logs.keys():
        plt.plot(model_logs[model_name]['score_steps'], model_logs[model_name]['scores'], label=model_name)
    plt.legend()

def show_best_steps(model_logs, types: list=['link', 'split']):
    bests = defaultdict(dict)
    for model_name in model_logs.keys():
        this_best_score = model_logs[model_name]['scores'][model_logs[model_name]['score_steps'] == model_logs[model_name]['best_step']][0]
        print(f'{model_name} \n\t best step: {model_logs[model_name]["best_step"]} \n\t with score {this_best_score}')
        
        type = get_model_type(model_name)
        if type not in bests.keys() or bests[type]['score'] > this_best_score:
            bests[type] = {'score': this_best_score, 'model_name': model_name, 'step': model_logs[model_name]["best_step"], 'layer_name': get_best_layer(type, '#INSERT SEED#', model_logs[model_name]["best_step"])}
        
    for type in types:
        print(f'Best {type}: \n\t model_name: {bests[type]["model_name"]} \n\t layer_name: {bests[type]["layer_name"]} \n\t score: {bests[type]["score"]}')

def get_best_layer(type, seed, step,
            net_raw_prefix='CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407',
            net_raw_suffix = '_netG2_184tCrp',
            type_dict = {
                'split': 'SplitNoBottle',
                'link': 'LinkNoBottle'
            }):
    return f'volumes/{net_raw_prefix}{type_dict[type]}_seed{seed}_checkpoint{step}{net_raw_suffix}'
# %%
