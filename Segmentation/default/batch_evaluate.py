#%%
from __future__ import print_function
from collections import defaultdict
import json
from io import StringIO
from jsmin import jsmin
from glob import glob
from shutil import copy
import os
import logging
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/Utils')
from make_foreground_mask import *
from wkw_seg_to_zarr import download_wk_skeleton
sys.path.append('/n/groups/htem/users/jlr54/raygun/Segmentation')
from rasterize_skeleton import *
import numpy as np
import daisy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#%%
def get_train_predict(folder, json_name):
    name = json_name.strip('/').split('/')[-1].replace('.json','')
    train = folder.strip('/').split('/')[-1]
    predict = name.replace('segment_', '').replace(train+'_', '')
    return train, predict

def batch_evaluate(seg_dict):   
    if seg_dict is None:
        raise ValueError('Automatic seg_dict generation not yet implemented.')
        # seg_dict = mk_seg_dict(src)

    metric_dict = {}
    save_dict = {}
    for train_dir, json_list in seg_dict.items():
        os.chdir(train_dir)
        for config in json_list:
            try:
                train, predict = get_train_predict(train_dir, config)
                metric_dict[train, predict] = rasterize_and_evaluate(config)
                save_dict[f'{train}_{predict}'] = metric_dict[train, predict]
            except:
                print(f'Failed in {train_dir} on {config}')
        os.chdir('../')

    try:
        os.makedirs('metrics', exist_ok =True)
        with open('metrics/test_thresh_metrics.json', 'w') as f:
            json.dump(save_dict, f)
    except:
        pass
    return metric_dict

#%%
#Should be run from folder where batch_train_affinities.py is
if __name__ == "__main__":
    train_dirs = glob(f'/{os.path.join(*__file__.split("/")[:-1], "train_*")}/')
    seg_dict = {}
    for train_dir in train_dirs:
        seg_dict[train_dir] = glob(os.path.join(train_dir, 'segment_train*.json'))

    batch_evaluate(seg_dict)
    print("All done.")
