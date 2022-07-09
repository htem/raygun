#%%
from datetime import datetime
from io import StringIO
import json
import os
from shutil import copy
from glob import glob
import logging
from subprocess import  Popen, check_output
import sys
from time import sleep

sys.path.append('/n/groups/htem/users/jlr54/raygun/Utils')
from wkw_seg_to_zarr import download_wk_skeleton

from jsmin import jsmin

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

# %%
def update_watchers(current_iter=None, exts=['local', 'sbatch']):
    folders = glob('train*')
    watchers = []
    for ext in exts:
        watchers += glob(f'default/network_watcher.{ext}')
    
    #Find most recent skeleton
    skel_file = get_updated_skeleton()

    for folder in folders:                
        with open(f'{folder}/segment.json', 'r') as file:
            segment_config = json.load(StringIO(jsmin(file.read())))
        segment_config['SkeletonConfig']['file'] = skel_file
        if current_iter is not None:
            segment_config['Network']['iteration'] = int(current_iter)
        with open(f"{folder}/segment.json", "w") as config_file:
            json.dump(segment_config, config_file)
        for watcher in watchers:
            logger.info(f'Updating {copy(watcher, folder)}...')
        
def get_updated_skeleton():
    with open('default/segment.json', 'r') as default_file:
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
    

def run_watchers(ext="local", current_iter=None, folders=None):
    command = {"local": "bash", "sbatch": "sbatch"}[ext]
    if current_iter is None:
        current_iter = kwargs["save_every"]
    
    if folders is None:
        folders = [watcher.split('/')[0] for watcher in glob(f'train*/network_watcher.{ext}')]

    jobs = {}
    for folder in folders:
        logger.info(f'Launching watcher for {folder}...') 
        os.chdir(folder)        
        with open('train_kwargs.json', 'r') as kwargs_file:
            kwargs = json.load(kwargs_file)
        command_str = f'{command} network_watcher.{ext} {kwargs["save_every"]} {kwargs["max_iteration"]} {current_iter} {os.getcwd()}/segment.json {kwargs["raw_ds"].split("/")[-1]}'
        success = os.system(command_str) == 0
        if success:
            job = 'Job run/submitted successfully: {folder}'
        else:
            job = 'Job run/submission failed: {folder}'

        os.chdir('../')
        jobs[folder] = job
    
    return jobs

def watch_watchers(jobs, ext='local'):
    status = {None: 'Running', 0: 'Success'}
    all_done = True
    out = [f'{datetime.now()}\n']
    for name, job in jobs.items():
        if ext == 'local':
            all_done = all_done and (job.poll() is not None)
            out.append(f'{name} (PID# {job.pid}) Status - {status.get(job.returncode, f"Error({job.returncode})")} \n')
        elif ext == 'sbatch':
            out.append(f'{name}: {job} \n')
    with open('watcher.jobs', 'w') as f:
        f.writelines(out)
    return all_done

# %%
if __name__ == '__main__':
    ext = 'local'
    current_iter = None
    folders = None
    if len(sys.argv) > 1:
        ext = sys.argv[1]
        if len(sys.argv) > 2:
            current_iter = sys.argv[2]
            if len(sys.argv) > 3:
                folders = sys.argv[3:]
    update_watchers(current_iter)
    jobs = run_watchers(ext, current_iter, folders)
    # all_done = False
    # while not all_done:
    #     all_done = watch_watchers(jobs, ext)
    #     sleep(10)
