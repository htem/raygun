#!/bin/bash

#SBATCH -p gpu_connectomics
#SBATCH --gres=gpu:1
#SBATCH --account=connectomics_contrib
#SBATCH -t 32:00:00 
#SBATCH -c 1
#SBATCH --mem=50GB
#SBATCH -o network_watcher.out

module load gcc/9.2.0
module load cuda/11.2

increment=$1
final_iteration=$2
curr_iteration=$3
config_json=$4

# venv
. /n/groups/htem/users/tmn7/envs/ubuntu18_python37/bin/activate
if [ "$PYTHONPATH" != */n/groups/htem/Segmentation/shared-nondev* ]; then
    echo INFO: PYTHONPATH env does not have segway... adding it
    export PYTHONPATH=$PYTHONPATH:/n/groups/htem/Segmentation/shared-nondev
fi

# . /n/groups/htem/Segmentation/shared-nondev/cbx_fn/segway2/gt_scripts/activate_gt_env.sh "$config_json";
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Add check so that the  next command only runs if previous command ran succesfully? 
# Wait until network is running
while [ ! -f .running ] && [ -f .done ]
do
    sleep 1m
done

while [ $((curr_iteration-increment)) -ne $final_iteration ]
do

    # Check if checkpoint exists
    checkpoint_file="model_checkpoint_$curr_iteration"
    if [ ! -f "$checkpoint_file" ]; then
        echo "$checkpoint_file does not exist"
        sleep 5m
        continue
    fi
    if [ -f "$checkpoint_file" ]; then
        echo "$checkpoint_file does exist"
    
        #TODO: Determine if this is necessary
        # # Remove existing zarr 
        # if [ -d "${config_json/json/"zarr"}" ]; then
        #     echo "Removing zarr"
        #     rm -r "${config_json/json/"zarr"}"
        # fi

        # Affinities
        python /n/groups/htem/Segmentation/shared-nondev/segway2/tasks/segmentation/task_01.py $config_json
        
        # Agglomerate
        # TODO Check for waterz error
        python /n/groups/htem/Segmentation/shared-nondev/segway2/tasks/segmentation/task_06a_extract_segments.py $config_json
        
        # Evaluate

        . /n/groups/htem/users/mkn5/venv/daisy-2/bin/activate
        python /n/groups/htem/users/mkn5/repos/jeff_scripts/rasterize_skeleton.py $config_json "./metrics/$curr_iteration.json" "./metrics/best_metric.json" $increment
        
        # Increment
        curr_iteration=$((curr_iteration+increment))
        echo  $curr_iteration
        sleep 1m
    fi
done

echo "all done"
