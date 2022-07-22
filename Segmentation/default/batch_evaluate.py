#%%
import json
from glob import glob
import os
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/Segmentation')
from rasterize_skeleton import *

#%%
def get_train_predict(folder, json_name):
    name = json_name.strip('/').split('/')[-1].replace('.json','')
    train = folder.strip('/').split('/')[-1]
    predict = name.replace('segment_', '').replace(train+'_', '')
    return train, predict

def get_seg_dict():
    # print(f'Finding training directories at: \n {os.path.join(os.path.dirname(__file__), "train_*")}')
    train_dirs = glob(f'{os.path.join(os.path.dirname(__file__), "train_*")}')
    seg_dict = {}
    for train_dir in train_dirs:
        seg_dict[train_dir] = glob(os.path.join(train_dir, 'segment_train*.json'))
    return seg_dict

def batch_evaluate(seg_dict=None, force_redo=False):   
    if not force_redo and os.path.exists('metrics/test_thresh_metrics.json'):
        with open('metrics/test_thresh_metrics.json', 'r') as f:
            save_dict = json.load(f)
        metric_dict = {}
        for key, metrics in save_dict.items():
            train = key[:key.find('predict')-1]
            predict = key[key.find('predict'):]
            metric_dict[train, predict] = metrics
        return metric_dict

    if seg_dict is None:
        seg_dict = get_seg_dict()
        # print(f'seg_dict = {seg_dict}')

    metric_dict = {}
    save_dict = {}
    for train_dir, json_list in seg_dict.items():
        os.chdir(train_dir)
        print(f'Evaluating in {train_dir}...')
        for config in json_list:
            try:
                train, predict = get_train_predict(train_dir, config)
                metric_dict[train, predict] = rasterize_and_evaluate(config)
                print(f'Evaluated {config}')
                save_dict[f'{train}_{predict}'] = metric_dict[train, predict]
            except:
                print(f'Failed on {config}')
        os.chdir('../')

    try:
        os.makedirs('metrics', exist_ok =True)
        with open('metrics/test_thresh_metrics.json', 'w') as f:
            json.dump(save_dict, f, indent=4)
    except:
        pass
    return metric_dict

#%%
#Should be run from folder where batch_train_affinities.py is
if __name__ == "__main__":
    batch_evaluate()
    print("All done.")
