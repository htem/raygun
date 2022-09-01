from __future__ import print_function
import json
from glob import glob
from distutils.dir_util import copy_tree
import os
import logging
import jax

from train_affinity import *

logging.basicConfig(level=logging.INFO)

n_devices = jax.local_device_count()
# n_devices = 1

#%%
def get_train_foldername(name, net_res={'netg1': '30nm', 'netg2': '90nm'}):
    #CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed13_checkpoint330000_netG2
    parts = name.lower().split('_')
    res = ''
    id = ['', '', '']
    source = ''
    i = -1
    while (res=='' or source=='') and abs(i) <= len(parts):
        part = parts[i]
        if 'nm' in part:
            res = part
        elif 'net' in part:
            res = net_res[part]
        elif 'interpolated' in part or 'raw' in part:
            source = 'real'
        elif 'checkpoint' in part:
            id[2] = part.replace('checkpoint', 'c')
        elif 'seed' in part:
            id[1] = part.replace('seed', 's')
        elif 'link' in part:
            id[0] = part[:part.find('link')].split('-')[0]
            source = 'link'
        elif 'split' in part:
            id[0] = part[:part.find('split')].split('-')[0]
            source = 'split'
        i -= 1
    return f'train_{source}{id[0]}{id[1]}{id[2]}_{res}'
#%%

if __name__ == "__main__":

    raw_ds_list = ['volumes/raw_30nm', 'volumes/interpolated_90nm_aligned', 'volumes/*_netG2']

    # if 'debug' in sys.argv:
    #     max_iteration = 10
    #     num_workers = 2
    #     cache_size = 1
    #     snapshot_every = 1
    # elif 'debug_perf' in sys.argv:
    #     max_iteration = 1000
    #     num_workers = 24
    #     snapshot_every = 10
    # else:
    #     try:
    #         max_iteration = int(sys.argv[1])
    #         num_workers = int(sys.argv[2])
    #     except:
    #         max_iteration = 300000
    #         num_workers = 16*n_devices
    #     cache_size = num_workers*2
    # batch_size = 1*n_devices

    with open('default/train_kwargs.json', 'r') as default_file:
        default_kwargs = json.load(default_file)
    
    temp = []
    for i, raw_ds in enumerate(raw_ds_list):
        if '*' in raw_ds:
           print(f'Collecting {raw_ds_list.pop(i)}...') 
           temp += glob(f'{default_kwargs["dense_samples"][0]}/{raw_ds}')
        else:
            temp.append(raw_ds)
    raw_ds_list = temp
        
    for raw_ds in raw_ds_list:
        kwargs = default_kwargs.copy()
        kwargs['raw_ds'] = raw_ds
        foldername = get_train_foldername(raw_ds.split('/')[-1])
        copy_tree('default', foldername)
        with open(f"{foldername}/train_kwargs.json", "w") as config_file:
            json.dump(kwargs, config_file)
        
        os.system(f'cd {foldername}')
        os.system('sbatch train.sbatch')
        os.system('cd ..')
