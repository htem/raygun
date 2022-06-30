#%% 
from __future__ import print_function
import json
from io import StringIO
from jsmin import jsmin
from glob import glob
from distutils.dir_util import copy_tree, remove_tree
import os
import logging
# import jax
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/Utils')
from make_foreground_mask import *
from wkw_seg_to_zarr import download_wk_skeleton
import daisy

logging.basicConfig(level=logging.INFO)

# n_devices = jax.local_device_count()
# n_devices = 1

#%%
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

def batch_train_affinities(raw_ds_list, seg_ds_dict, raw_srcs=None):
    with open('default/train_kwargs.json', 'r') as default_file:
        default_kwargs = json.load(default_file)

    if raw_srcs is None:
        check_raws = False
        raw_srcs = default_kwargs["dense_samples"]
    else:
        check_raws = True
        
    with open('default/segment.json', 'r') as default_file:
        segment_config = json.load(StringIO(jsmin(default_file.read())))

    temp = []
    for i, raw_ds in enumerate(raw_ds_list):
        if '*' in raw_ds:
            print(f'Collecting {raw_ds}...') 
            raw_list = glob(f'{raw_srcs[0]}/{raw_ds}')            
            raw_list = [raw.split('/')[-(raw_ds.count('/')+1):] for raw in raw_list]
            temp += [f'{raw[0]}/{raw[1]}' for raw in raw_list]
        else:
            temp.append(raw_ds)
    raw_ds_list = temp
    
    if check_raws:
        raw_names = set([raw.replace('volumes/', '') for raw in raw_ds_list])    
        for raw_src, dense_sample in zip(raw_srcs, default_kwargs["dense_samples"]):
            current_raws = set([ds.split('/')[-1] for ds in glob(dense_sample+'/volumes/*')])        
            missing = ['volumes/'+ missed for missed in set(raw_names).difference(current_raws)]
            if len(missing) > 0:
                print(f'Adding missing raws: {missing}')
                add_raws_to_zarr(raw_src, dense_sample, missing)

    #check if any files need making
    if default_kwargs['unlabeled_mask_ds'] == '':
        labels_ds = default_kwargs['labels_ds']
        for source_file in default_kwargs['dense_samples']:
            if not os.path.exists(f'{source_file}/{labels_ds}_foreground_mask'):
                make_foreground_mask(source_file, labels_ds)
        default_kwargs['unlabeled_mask_ds'] = f'{labels_ds}_foreground_mask'
        with open('default/train_kwargs.json', 'w') as default_file:
             json.dump(default_kwargs, default_file)
    
    #get skeleton for validation
    if not os.path.exists(segment_config['SkeletonConfig']['file']):
        files = glob('./skeletons/*')
        if len(files) == 0 or segment_config['SkeletonConfig']['file'] == 'update':
            segment_config['SkeletonConfig']['file'] = download_wk_skeleton(
                                            segment_config['SkeletonConfig']['url'], 
                                            os.getcwd() + '/skeletons/',
                                            overwrite=True
                                        )
        else:
            segment_config['SkeletonConfig']['file'] = max(files, key=os.path.getctime)

    for raw_ds in raw_ds_list:
        kwargs = default_kwargs.copy()
        kwargs['raw_ds'] = raw_ds
        foldername = get_train_foldername(raw_ds.split('/')[-1])
        def setup():
            os.chdir(f'{foldername}/')
            with open(f"train_kwargs.json", "w") as kwargs_file:
                json.dump(kwargs, kwargs_file)
            segment_config['Input']['raw_dataset'] = seg_ds_dict[foldername.split('_')[-1]]
            segment_config['Network']['name'] = foldername
            segment_config['Network']['train_dir'] = os.getcwd()
            with open(f"segment.json", "w") as config_file:
                json.dump(segment_config, config_file)
            os.system('sbatch train.sbatch')
            os.system(f'bash network_watcher.sh \
                {kwargs["save_every"]} \
                {kwargs["max_iteration"]} \
                {kwargs["save_every"]} \
                segment.json')
            os.chdir('../')

        if len(glob(foldername)) == 0: 
            print(f'Training affinity prediction on {raw_ds}.')
            copy_tree('default', foldername)
            setup()

        elif len(glob(f'{foldername}/.running')) == 0 and len(glob(f'{foldername}/.done')) == 0:
            print(f'Training affinity prediction on {raw_ds}.')            
            setup()


        elif len(glob(f'{foldername}/.done')) > 0 and len(glob(f'{foldername}/checkpoints/{raw_ds.split("/")[-1]}_checkpoint_{kwargs["max_iteration"]}')) == 0:
            print(f'Retraining affinity prediction on {raw_ds}, because it did not finish last time.')
            remove_tree(foldername)
            copy_tree('default', foldername)
            setup()

        else:
            if len(glob(f'{foldername}/.running')) > 0:
                print(f'Skipping {raw_ds} because it is currently training.')
            elif len(glob(f'{foldername}/.done')) > 0:
                print(f'Skipping {raw_ds} because it has already trained.')

        os.symlink(f'{os.getcwd()}/{foldername}/log/{raw_ds.split("/")[-1]}', f'{os.getcwd()}/logs/{foldername}', target_is_directory=True)

#%%
#Should be run from folder where batch_train_affinities.py is
if __name__ == "__main__":

    raw_ds_list = ['volumes/raw_30nm', 
                    'volumes/interpolated_90nm_aligned',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed3_checkpoint340000_netG2_184tCrp',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed13_checkpoint350000_netG2_184tCrp',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed4_checkpoint310000_netG2_184tCrp',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed42_checkpoint340000_netG2_184tCrp',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed13_checkpoint330000_netG2_184tCrp',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed42_checkpoint310000_netG2_184tCrp',
                    ]    

    seg_ds_dict = {
        '30nm': 'volumes/raw_30nm',
        '90nm': 'volumes/interpolated_90nm_aligned'
    }

    raw_srcs = ["/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5"]

    batch_train_affinities(raw_ds_list, seg_ds_dict, raw_srcs)