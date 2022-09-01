#%% 
from __future__ import print_function
import json
from io import StringIO
from jsmin import jsmin
from glob import glob
from distutils.dir_util import copy_tree, remove_tree
import os
import logging
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/Utils')
import daisy

logging.basicConfig(level=logging.INFO)

#%%
def add_raws_to_zarr(src, dest, datasets, num_workers=34, chunk_wise=False):
    for ds in datasets:
        if os.path.isdir(os.path.join(dest, ds)):
            remove_tree(os.path.join(dest, ds))
        from_ds = daisy.open_ds(src, ds)
        to_ds = daisy.prepare_ds(dest,
                                ds,
                                from_ds.roi,
                                from_ds.voxel_size,
                                from_ds.dtype,
                                )
        
        if chunk_wise:
            rw_roi = daisy.Roi((0,0,0), from_ds.voxel_size * 64)                
            #Write data to new dataset
            try: #New daisy            
                def save_chunk(block:daisy.Roi):
                    to_ds.__setitem__(block.write_roi, from_ds.to_ndarray(block.read_roi))
                task = daisy.Task(
                    f'{ds}---add_missing',
                    from_ds.roi,
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
                        to_ds.__setitem__(block.write_roi, from_ds.to_ndarray(block.read_roi))
                        return 0 # success
                    except:
                        return 1 # error
                success = daisy.run_blockwise(from_ds.roi,
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
        else:
            print(f'Copying {ds} non-chunkwise - this may take a bit as it is {from_ds.shape}...')
            to_ds[from_ds.roi] = from_ds.to_ndarray()

def check_raws(raw_ds_list, raw_srcs, force=False):
    with open('default/train_kwargs.json', 'r') as default_file:
        default_kwargs = json.load(default_file)
    
    print('Gathering raw list...')
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
    
    if not force:
        print('Finding missing...')
        raw_names = set([raw.replace('volumes/', '') for raw in raw_ds_list])    
        for raw_src, dense_sample in zip(raw_srcs, default_kwargs["dense_samples"]):
            current_raws = set([ds.split('/')[-1] for ds in glob(dense_sample+'/volumes/*')])        
            missing = ['volumes/'+ missed for missed in set(raw_names).difference(current_raws)]
            if len(missing) > 0:
                print(f'Adding missing raws: {missing}')
                print(f'From {raw_src}...')
                add_raws_to_zarr(raw_src, dense_sample, missing)
    elif force:
        raw_names = set([raw.replace('volumes/', '') for raw in raw_ds_list])    
        for raw_src, dense_sample in zip(raw_srcs, default_kwargs["dense_samples"]):
            print(f'Adding raws: {raw_ds_list}')
            print(f'From {raw_src}...')
            add_raws_to_zarr(raw_src, dense_sample, raw_ds_list)

    #check if any files need making
    if default_kwargs['unlabeled_mask_ds'] == '':
        labels_ds = default_kwargs['labels_ds']
        for source_file in default_kwargs['dense_samples']:
            if not os.path.exists(f'{source_file}/{labels_ds}_foreground_mask'):
                make_foreground_mask(source_file, labels_ds)
        default_kwargs['unlabeled_mask_ds'] = f'{labels_ds}_foreground_mask'
    
#%%

#Should be run from folder where batch_train_affinities.py is
if __name__ == "__main__":
    force = 'force' in sys.argv
    raw_ds_list = ['volumes/raw_30nm', 
                    'volumes/interpolated_90nm_aligned',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed4_checkpoint310000_netG2_184tCrp',
                    'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed42_checkpoint340000_netG2_184tCrp'
                    ]    

    raw_srcs = ["/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5"]

    print('Checking raws...')
    check_raws(raw_ds_list, raw_srcs, force)
    print('Done!')