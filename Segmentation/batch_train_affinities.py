from __future__ import print_function
import sys
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/gunpowder-1.2.2-220114')
# sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/gunpowder.210911')
# sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/gunpowder-1.3')
from gunpowder.jax import Train as JaxTrain

import os
from gunpowder import *
from reject import Reject
import os
import math
import json
from glob import glob
import numpy as np
import logging
from mknet import create_network
import jax

from train_affinity import *

global cache_size
global snapshot_every
cache_size = 40
snapshot_every = 1000

logging.basicConfig(level=logging.INFO)

n_devices = jax.local_device_count()
# n_devices = 1

dense_samples = [
   '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5',
]

raw_ds_list = ['volumes/raw_30nm']
labels_ds = 'volumes/gt_CBvBottomGT_training_0_BrianReicher_20220521_close-open_r2'
labels_mask_ds = 'volumes/trainingMask_center400'
unlabeled_mask_ds = 'volumes/gt_CBvBottomGT_training_0_BrianReicher_20220521_close-open_r2_foreground_mask'

DefaultConfig = "/n/groups/htem/Segmentation/shared-nondev/segway2/tasks/segmentation/default_config_cbx_xray.json"

def make_gt_json(raw_name):
    with open(DefaultConfig, 'r') as f:
        config = json.load(f)
    
    ...

if __name__ == "__main__":

    if 'debug' in sys.argv:
        # iteration = 5
        # num_workers = 6
        # cache_size = 5
        # snapshot_every = 1
        max_iteration = 10
        num_workers = 2
        cache_size = 1
        snapshot_every = 1
    elif 'debug_perf' in sys.argv:
        max_iteration = 1000
        num_workers = 24
        # num_workers = 36
        snapshot_every = 10
    else:
        try:
            max_iteration = int(sys.argv[1])
            num_workers = int(sys.argv[2])
        except:
            max_iteration = 100000
            num_workers = 16*n_devices
            # cache_size = 24*n_devices
            # num_workers = 24

        # cache_size = 40*batch_size
        cache_size = num_workers*2
        # cache_size = 1
        # cache_size = 80

    batch_size = 1*n_devices
    
    temp = []
    for i, raw_ds in enumerate(raw_ds_list):
        if '*' in raw_ds:
           print(f'Collecting {raw_ds_list.pop(i)}...') 
           temp += glob(f'{dense_samples[0]}/{raw_ds}')
        else:
            temp.append(raw_ds)
    raw_ds_list = temp

        
    for raw_ds in raw_ds_list:
        train_affinity(dense_samples,
                raw_ds,
                labels_ds,
                labels_mask_ds,
                unlabeled_mask_ds,
                max_iteration,
                num_workers,
                batch_size)
        
        make_gt_json(f'{raw_ds.split("/")[-1]}')
