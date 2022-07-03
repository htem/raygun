from functools import partial
import json
import daisy
import sys
import os
import json
sys.path.append('/n/groups/htem/Segmentation/shared-nondev/cbx_fn/segway2/gt_scripts')
from skeleton import parse_skeleton
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev')
import gt_tools
import numpy as np
from skimage.draw import line_nd
from funlib import evaluate

def nm2px(coord, voxel_size, offset):
    # removes offset and converts to px
    return [int((a/b)-(c/b)) for a,b,c in zip(coord, voxel_size, offset)]

def rasterize_and_evaluate(config, cube_size=1024):

    # Skel=tree_id:[Node_id], nodes=Node_id:{x,y,z}
    skeletons, nodes = parse_skeleton(config['SkeletonConfig'])

    # Cardiac arrest is served
    # {Tree_id:[[xyz]]}
    skel_zyx={ tree_id: [nodes[nid]['zyx'] for nid in node_id] for tree_id,node_id in skeletons.items()}

    # load segmentation
    segment_dataset = config["segment_ds"]
    segment_file = config["Input"]["output_file"]
    segment_ds = daisy.open_ds(segment_file, segment_dataset)
    segment_array = segment_ds[segment_ds.roi].to_ndarray()

    # Initialize rasterized skeleton image
    # config['roi_shape'] + 2 * config['roi_offset']
    image=np.zeros((cube_size,)*3,dtype=np.uint)

    px = partial(nm2px, voxel_size=config["SkeletonConfig"]["voxel_size_xyz"], offset=config["Input"]["roi_offset"])
    for id,tree in skel_zyx.items():
    #iterates through ever node and assigns id to {image}
        for i in range(0,len(tree)-1,2):
            line = line_nd(px(tree[i]), px(tree[i+1]))    
            image[line] = int(list(skel_zyx.keys())[5])

    # Metrics
    pad = (cube_size - np.array(segment_array.shape)) // 2
    return evaluate.rand_voi(segment_array,image[pad[0]:-pad[0], pad[1]:-pad[1], pad[2]:-pad[2]])

def nvi_score(eval):
    return np.sqrt(eval["nvi_split"]*eval["nvi_merge"])

if __name__=="__main__":

    config_file = sys.argv[1]
    increment = int(sys.argv[2])
    METRIC_OUT_JSON = "./metrics/metrics.json"
    BEST_METRIC_JSON = "./metrics/best.iteration"

    config = gt_tools.load_config(config_file)
    current_iteration = config["Network"]["iteration"]
    evaluation = rasterize_and_evaluate(config)

    #check append
    if not os.path.isfile(METRIC_OUT_JSON):
        metrics = {current_iteration: evaluation}
    else:        
        with open(METRIC_OUT_JSON,'r') as f:
            metrics = json.load(f)
        metrics[current_iteration] = evaluation
    with open(METRIC_OUT_JSON,'w') as f:
        json.dump(metrics, f)

    # Save best 
    if not os.path.isfile(BEST_METRIC_JSON):
        with open(BEST_METRIC_JSON, 'w') as f:
            json.dump({current_iteration: evaluation})
    else:
        with open(BEST_METRIC_JSON, 'r') as f:
            best_metric = json.load(f)
        curr_best = list(best_metric.values())[0]
        
        if nvi_score(curr_best) > nvi_score(evaluation):
            with open(BEST_METRIC_JSON, 'w') as f:
                json.dump({current_iteration: evaluation})

    # Increment config
    # print(config_file)
    with open(config_file,'r+') as f:
        config=json.loads(f.read())

    # Save config file
    with open(config_file,'w+') as f:       
        config["Network"]["iteration"] = current_iteration + increment
        json.dump(config, f, indent=4)
        