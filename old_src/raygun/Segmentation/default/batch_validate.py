#%%
import json
import os
from shutil import copy
from glob import glob
import logging
import sys
import hashlib 

sys.path.append('/n/groups/htem/users/jlr54/raygun/Utils')
from wkw_seg_to_zarr import download_wk_skeleton
sys.path.append('/n/groups/htem/users/jlr54/raygun/Segmentation')
from rasterize_skeleton import *
from batch_segment import *

from jsmin import jsmin

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

#%%
#FIX VALIDATION
default_config_fn='segment.json'
best_iter_fp='metrics/best.iteration'
train_dirs = glob(os.path.join(os.getcwd(), 'train_*s*'))
configs = defaultdict(list)
for train_dir in train_dirs:
    copy('default/validator.local', os.path.join(train_dir, 'validator.local'))
    os.chdir(train_dir)
    with open(default_config_fn, 'r') as f:
        old_config = json.load(f)
    with open('train_kwargs.json', 'r') as f:
        kwargs = json.load(f)
    with open(f"{train_dir}/{best_iter_fp}", 'r') as f: # Set to best iteration
        best = json.load(f)

    raw_ds = kwargs['raw_ds']
    new_config = old_config.copy()
    new_config['Input']['raw_dataset'] = raw_ds
    new_config['Network']['iteration'] = int(list(best.keys())[0])
    raw_name = raw_ds.strip('/').split('/')[-1]
    train_name = old_config['Network']['name']
    predict_name = get_predict_name(train_name, raw_name)
    new_config['Input']['raw_dataset'] = raw_ds
    new_config['Input']['db_name'] = hashlib.sha1(str(train_name+predict_name).encode('utf-8')).hexdigest()
    config_name = f"{train_dir}/segment_{predict_name}.json"
    with open(config_name, "w") as config_file: #Save config
        json.dump(new_config, config_file)
    configs[train_dir].append(config_name)
    success = os.system(f"bash validator.local {config_name}")==0
    if success:
        print(f'Evaluated {config_name}!')
    else:
        print(f'Failed to evaluate {config_name} =(')
    os.chdir('../')

#%%
fix_list = [
                '/n/groups/htem/Segmentation/networks/xray_setups/eccv-bic/setup01/train_split20220407s3c340000_90nm/segment_train_split20220407s3c340000_90nm_predict_real_90nm.json',
                '/n/groups/htem/Segmentation/networks/xray_setups/eccv-bic/setup01/train_split20220407s42c340000_90nm/segment_train_split20220407s42c340000_90nm_predict_real_90nm.json'
                ]

for config in fix_list:
    metric_path = os.path.join('/', *config.split('/')[:-1], 'metrics/metrics.json')
    new_metric_name = config.replace('segment_', '').replace('.json', '')
    with open(metric_path, 'r') as f:
        metrics = json.load(f)
    
    new_metric = rasterize_and_evaluate(config, thresh_list=False)
    segment_ds = list(new_metric.keys())[0]
    new_metric['segment_ds'] = segment_ds

    metrics[new_metric_name] = new_metric[segment_ds]    

    with open(metric_path, 'w') as f:
        json.dump(metrics, f, indent=4)
# %%
