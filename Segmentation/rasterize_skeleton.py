#%%
from functools import partial
from glob import glob
import json
import daisy
import sys
import os
import json
sys.path.append('/n/groups/htem/Segmentation/shared-nondev/cbx_fn/segway2/gt_scripts')
from skeleton import parse_skeleton
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev')
import gt_tools
import zarr
import numpy as np
from skimage.draw import line_nd
from funlib import evaluate

def nm2px(coord, voxel_size, offset):
    # removes offset and converts to px
    return [int((a/b)-(c/b)) for a,b,c in zip(coord, voxel_size, offset)]

def rasterize_and_evaluate(config, cube_size=1024, thresh_list='volumes/segmentation_*'):
    if isinstance(config, str):
        config = gt_tools.load_config(config)

    # Skel=tree_id:[Node_id], nodes=Node_id:{x,y,z}
    skeletons, nodes = parse_skeleton(config['SkeletonConfig'])

    # Cardiac arrest is served
    # {Tree_id:[[xyz]]}
    skel_zyx={ tree_id: [nodes[nid]['zyx'] for nid in node_id] for tree_id,node_id in skeletons.items()}

    # Initialize rasterized skeleton image
    # config['roi_shape'] + 2 * config['roi_offset']
    image = np.zeros((cube_size,)*3,dtype=np.uint)

    px = partial(nm2px, voxel_size=config["SkeletonConfig"]["voxel_size_xyz"], offset=config["Input"]["roi_offset"])
    for id, tree in skel_zyx.items():
    #iterates through ever node and assigns id to {image}
        for i in range(0,len(tree)-1,2):
            line = line_nd(px(tree[i]), px(tree[i+1]))    
            image[line] = id
    
    # #Save GT rasterization
    # with zarr.open('segment.zarr', 'w') as f:
    #     f['skel_image/gt'] = image
    #     f['skel_image/gt'].attrs['resolution'] = config["SkeletonConfig"]["voxel_size_xyz"]
    #     f['skel_image/gt'].attrs['offset'] = tuple(config["Input"]["roi_offset"] - pad * daisy.Coordinate(config["SkeletonConfig"]["voxel_size_xyz"]))

    # load segmentation
    segment_file = config["Input"]["output_file"]
    if thresh_list is False:
        segment_datasets = [config["segment_ds"]]
    elif isinstance(thresh_list, str):
        segment_datasets = [os.path.join(*ds.strip('/').split('/')[-2:]) for ds in glob(os.path.join(segment_file, thresh_list))]
    else:
        segment_datasets = thresh_list

    evaluation = {}
    for segment_dataset in segment_datasets:
        segment_ds = daisy.open_ds(segment_file, segment_dataset)
        segment_array = segment_ds[segment_ds.roi].to_ndarray()
        pad = daisy.Coordinate(cube_size - np.array(segment_array.shape)) // 2
        evaluation[segment_dataset] = evaluate.rand_voi(image[pad[0]:-pad[0], pad[1]:-pad[1], pad[2]:-pad[2]], segment_array)

    return evaluation

def get_score(metrics, keys=['nvi_split', 'nvi_merge']):
        score = 0
        for key in keys:
            if not np.isnan(metrics[key]):
                score += metrics[key]
                # if metrics[key] != 0: #Discard any 0 metrics as flawed(?)
                #     score *= metrics[key]
            else:
                return 999
        return score

#%%
if __name__=="__main__":
    config_file = sys.argv[1]
    thresh_list = 'volumes/segmentation_*'
    update_best = False
    if len(sys.argv) > 2:
        if sys.argv[2] == 'update_best':
            update_best = True
            increment = config_file.strip('/').split('/')[-1].replace('segment_', '').replace('.json', '')
        else:
            increment = int(sys.argv[2])
    else:
        increment = config_file.strip('/').split('/')[-1].replace('segment_', '').replace('.json', '')
        thresh_list = False

    METRIC_OUT_JSON = "./metrics/metrics.json"
    BEST_METRIC_JSON = "./metrics/best.iteration"

    config = gt_tools.load_config(config_file)
    current_iteration = int(config["Network"]["iteration"])
    print(f'Evaluating {config_file} at iteration {current_iteration}...')
    evaluation = rasterize_and_evaluate(config, thresh_list=thresh_list)
    best_eval = {}
    for thresh, metrics in evaluation.items():
        if len(best_eval)==0 or get_score(best_eval[current_iteration]) > get_score(metrics):
            best_eval = metrics
            best_eval['segment_ds'] = thresh
            best_eval['iteration'] = current_iteration

    #check append
    if not os.path.isfile(METRIC_OUT_JSON):
        metrics = {current_iteration: evaluation}
    else:        
        with open(METRIC_OUT_JSON,'r') as f:
            metrics = json.load(f)
        if isinstance(increment, str) and not update_best: #for evaluating best threshold/iteration on different raw_datasets
            # best_eval[current_iteration]['iteration'] = current_iteration
            # metrics[increment] = best_eval[current_iteration]
            evaluation['iteration'] = current_iteration
            metrics[increment] = evaluation
        else:
            metrics[current_iteration] = evaluation
    with open(METRIC_OUT_JSON,'w') as f:
        json.dump(metrics, f, indent=4)

    # Increment config
    if update_best:
        print(f'New best = {best_eval}')            
        with open(BEST_METRIC_JSON, 'w') as f:
            json.dump(best_eval, f, indent=4)
    elif increment is not None and not isinstance(increment, str):
        # Save best 
        if not os.path.isfile(BEST_METRIC_JSON):
            with open(BEST_METRIC_JSON, 'w') as f:
                json.dump(best_eval, f, indent=4)
        else:
            with open(BEST_METRIC_JSON, 'r') as f:
                curr_best = json.load(f)
            
            if get_score(curr_best) > get_score(best_eval):
                print(f'New best = {best_eval}')            
                with open(BEST_METRIC_JSON, 'w') as f:
                    json.dump(best_eval, f, indent=4)

        # print(config_file)
        with open(config_file,'r+') as f:
            config=json.loads(f.read())

        # Save config file
        with open(config_file,'w+') as f:       
            config["Network"]["iteration"] = current_iteration + increment
            json.dump(config, f, indent=4)
        
# %%
