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

def update_segmentors(seg_dict, 
                    exts=['local', 'sbatch'], 
                    default_config_fp='default/segment_test.json', 
                    best_iter_fp='metrics/best.iteration',
                    ):
    segmentors = []
    for ext in exts:
        segmentors += glob(f'default/segmentor.{ext}')
    
    #Find most recent skeleton
    skel_file = get_updated_skeleton(default_config_fp)
    
    configs = defaultdict(list)
    for train_dir, raw_ds_list in seg_dict.items():
        train_name = train_dir.strip('/').split('/')[-1]
        with open(f"{default_config_fp}", 'r') as file:
            default_config = json.load(StringIO(jsmin(file.read())))

        src = default_config['Input']['raw_file']
        seg_dict[train_dir] = get_raws(raw_ds_list, src) 
        
        default_config['SkeletonConfig']['file'] = skel_file # Add new skeleton file

        with open(f"{train_dir}/{best_iter_fp}", 'r') as f: # Set to best iteration
            best = json.load(f)
        # if len(default_config['Input']['db_name']) >= 64: #TODO:FIX THIS IN SEGWAY
        #     default_config['Input']['db_name'] = default_config['Input']['db_name'][:63]
        default_config['Network']['iteration'] = int(best['iteration'])
        #Update other params
        default_config['Network']['name'] = train_name
        default_config['Network']['train_dir'] = train_dir
        default_config['segment_ds'] = best['segment_ds']

        #Make json for each dataset to segment with network
        for raw_ds in seg_dict[train_dir]:
            config = default_config.copy()
            raw_name = raw_ds.split('/')[-1]
            predict_name = get_predict_name(train_name, raw_name)
            config['Input']['raw_dataset'] = raw_ds
            config_name = f"{train_dir}/segment_{predict_name}.json"
            with open(config_name, "w") as config_file: #Save config
                json.dump(config, config_file, indent=4)
            configs[train_dir].append(config_name)

        for segmentor in segmentors: #Copy worker scripts
            logger.info(f'Updating {copy(segmentor, train_dir)}...')
    return configs
        
def get_updated_skeleton(default_config_fn='segment_test.json'):
    with open(default_config_fn, 'r') as default_file:
        segment_config = json.load(StringIO(jsmin(default_file.read())))
        
    if not os.path.exists(segment_config['SkeletonConfig']['file']):
        files = glob('./skeletons/*')
        if len(files) == 0 or segment_config['SkeletonConfig']['file'] == 'update':
            skel_file = download_wk_skeleton(
                        segment_config['SkeletonConfig']['url'].split('/')[-1], 
                        f'{os.getcwd()}/skeletons/',
                        overwrite=True,
                    )
        else:
            skel_file = max(files, key=os.path.getctime)
    skel_file = os.path.abspath(skel_file)
    
    return skel_file

def add_raws_to_zarr(src, dest, datasets, num_workers=34):
    for ds in datasets:
        this = daisy.open_ds(src, ds)
        that = daisy.prepare_ds(dest,
                                ds,
                                this.roi,
                                this.voxel_size,
                                this.dtype,
                                )
        rw_roi = daisy.Roi((0,0,0), this.voxel_size * 64)
                
        #Write data to new dataset
        try: #New daisy            
            def save_chunk(block:daisy.Roi):
                that.__setitem__(block.write_roi, this.__getitem__(block.read_roi))
            task = daisy.Task(
                f'{ds}---add_missing',
                this.roi,
                rw_roi,
                rw_roi,
                process_function=save_chunk,
                read_write_conflict=False,
                # fit='shrink',
                num_workers=num_workers,
                max_retries=2)
            success = daisy.run_blockwise([task])
        except: #Try old daisy
            def save_chunk(block:daisy.Roi):
                try:
                    that.__setitem__(block.write_roi, this.__getitem__(block.read_roi))
                    return 0 # success
                except:
                    return 1 # error
            success = daisy.run_blockwise(this.roi,
                rw_roi,
                rw_roi,
                process_function=save_chunk,
                read_write_conflict=False,
                # fit='shrink',
                num_workers=num_workers,
                max_retries=2)
        if success:
            print(f'Added {ds}.')
        else:
            print(f'Failed to save {ds}.')

def get_predict_name(train_name, raw_name, net_res={'netg1': '30nm', 'netg2': '90nm'}):
    #CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed13_checkpoint330000_netG2
    parts = raw_name.lower().split('_')
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
    return f'{train_name}_predict_{source}{id[0]}{id[1]}{id[2]}_{res}'

def get_raws(raw_ds_list, src):
    out = []
    for raw_ds in raw_ds_list:
        if '*' in raw_ds:
            print(f'Collecting {raw_ds}...') 
            raw_list = glob(f'{src}/{raw_ds}')            
            raw_list = [raw.split('/')[-(raw_ds.count('/')+1):] for raw in raw_list]
            out += [f'{raw[0]}/{raw[1]}' for raw in raw_list]
        else:
            out.append(raw_ds)
    return out

def batch_segment(seg_dict, ext='local'):
    command = {"local": "bash", "sbatch": "sbatch"}[ext]

    if seg_dict is None:
        raise ValueError('Automatic seg_dict generation not yet implemented.')
        # seg_dict = mk_seg_dict(src)

    configs = update_segmentors(seg_dict)
    for train_dir, config_list in configs.items():
        os.chdir(train_dir)
        for config in config_list:
            success = os.system(f"{command} segmentor.{ext} {config}")==0
            if success:
                print(f'Evaluated {config}!')
            else:
                print(f'Failed to evaluate {config} =(')
        os.chdir('../')


#%%
#Should be run from folder where batch_train_affinities.py is
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ext = sys.argv[1].lower()
    else:
        ext = 'local'

    res_dict = {
        '30nm': ['volumes/raw_30nm', 
                    'volumes/interpolated_90nm_aligned',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed4_checkpoint310000_netG1_184tCrp',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed42_checkpoint340000_netG1_184tCrp'
                    #TODO: Fill these in from segment_test.json (switch netG2-->netG1)                    
                    ],
        '90nm': ['volumes/interpolated_90nm_aligned']
    }

    train_dirs = glob(f'/{os.path.join(*__file__.split("/")[:-1], "train_*")}/')
    seg_dict = {}
    for train_dir in train_dirs:
        seg_dict[train_dir] = res_dict[train_dir.strip('/').split('/')[-1].split('_')[-1]]

    batch_segment(seg_dict, ext)
    print("All done.")
