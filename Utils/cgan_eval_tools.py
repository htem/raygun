#%%
from collections import defaultdict
from importlib.machinery import SourceFileLoader
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob
import os
import sys

import daisy
import torch
sys.path.insert(0, '/n/groups/htem/users/jlr54/gunpowder/')
from gunpowder import gunpowder as gp

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
def try_patch(cycleGun, side='A', pad=0, mode='eval', side_length=512, real=None, batch=None):
    if real is None:
        if batch is None:
            batch = cycleGun.test_prediction(side.upper(), side_length=side_length, cycle=False)

        real = batch[getattr(cycleGun, f'real_{side}')].data * 2 - 1
        real = real.squeeze()

    else:
        real = np.expand_dims(real, axis=0)

    if side.upper() == 'A':
        net = cycleGun.model.netG1.to('cuda')
    else:
        net = cycleGun.model.netG2.to('cuda')
            
    if mode.lower() == 'eval':
        # cycleGun.model.eval()
        net.eval()
    elif mode.lower() == 'real':
        return real, []
    else:
        # cycleGun.model.train()
        net.train()
    
    mid = real.shape[-1] // 2
    # test = net(torch.cuda.FloatTensor(real).unsqueeze(0))
    # pad = (real.shape[-1] - test.shape[-1]) // 2

    patch1 = torch.cuda.FloatTensor(real[:, :mid+pad, :mid+pad]).unsqueeze(0)
    patch2 = torch.cuda.FloatTensor(real[:, mid-pad:, :mid+pad]).unsqueeze(0)
    patch3 = torch.cuda.FloatTensor(real[:, :mid+pad, mid-pad:]).unsqueeze(0)
    patch4 = torch.cuda.FloatTensor(real[:, mid-pad:, mid-pad:]).unsqueeze(0)

    patches = [patch1, patch2, patch3, patch4]
    fakes = []
    for patch in patches:
        test = net(patch)
        fakes.append(test.detach().cpu().squeeze())

    if pad != 0:
        fake_comb = torch.cat((torch.cat((fakes[0][:-pad, :-pad], fakes[1][pad:, :-pad])), torch.cat((fakes[2][:-pad, pad:], fakes[3][pad:, pad:]))), axis=1)
    else:
        fake_comb = torch.cat((torch.cat((fakes[0], fakes[1])), torch.cat((fakes[2], fakes[3]))), axis=1)
    
    return fake_comb.squeeze(), real.squeeze()

def get_center_roi(ds, side_length, ndims, voxel_size=None):
    if voxel_size is None:
        voxel_size = ds.voxel_size
    shape = list((1,)*3)
    shape[:ndims] = (side_length * 2,)*ndims
    real_shape = voxel_size * daisy.Coordinate(shape)
    roi = daisy.Roi(ds.roi.center, real_shape)
    roi = roi.shift(-real_shape / 2).snap_to_grid(ds.voxel_size)
    return roi

def get_real(cycleGun, side, roi=None):
    ds = daisy.open_ds(getattr(cycleGun, f'src_{side}'), getattr(cycleGun, f'{side}_name'))
    if roi is None:
        roi = get_center_roi(ds, cycleGun.side_length, cycleGun.ndims, daisy.Coordinate(cycleGun.common_voxel_size))    

    data = ds.to_ndarray(roi)
    if len(data.shape) > cycleGun.ndims:
        data = data[...,0]
    return (data / 255) * 2 - 1

def show_patches(cycleGun, pad=0):
    if isinstance(cycleGun, str):
        path = cycleGun
        sys.path.insert(0, path)
        train = SourceFileLoader('train', f'{path}train.py').load_module()
        sys.path.pop(0)
        cycleGun = train.cycleGun
        del train
        del sys.modules['train']
        cycleGun.load_saved_model()

    side_length = cycleGun.side_length
    fig, axs = plt.subplots(2, 3, figsize=(30,20))
    reals = []
    print(f'{cycleGun.model_name} at iteration {cycleGun.iteration}')
    for i, side in enumerate(['A', 'B']):
        axs[i,0].set_ylabel(side)
        # batch = cycleGun.test_prediction(side.upper(), side_length=side_length*2, cycle=False)
        real = get_real(cycleGun, side)
        reals.append(real)
        for j, mode in enumerate(['real', 'eval', 'train']):
            if mode == 'real':
                img = real
            else:
                img, _ = try_patch(cycleGun, side=side, mode=mode, pad=pad, side_length=side_length, real=real)
            axs[i,j].imshow(img, cmap='gray', vmin=-1, vmax=1)
            axs[i,j].set_title(mode)
    return cycleGun, fig, real


#%%
pad = 10
base = '/n/groups/htem/ResolutionEnhancement/cycleGAN_setups/set20220729/'
# cycleGun, fig, real = show_patches(base+'resnet_track0001/', pad=pad)
# cycleGun, fig, real = show_patches(base+'resnet_track001/', pad=pad)
# cycleGun, fig, real = show_patches(base+'resnet_track01/', pad=pad)
cycleGun, fig, real = show_patches(base+'unet/', pad=pad)

#%%
sys.path.append(base+'/resnet_track001/')
from train import *
cycleGun.load_saved_model(base+'/resnet_track001/models/cycleGAN_setups_set20220728_resnet_track001_checkpoint_100000')
#%%
cycleGun, fig, real = show_patches(cycleGun)

# %%
