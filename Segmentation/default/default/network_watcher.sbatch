#!/bin/bash

#SBATCH -p gpu_connectomics
#SBATCH --gres=gpu:1
#SBATCH --account=connectomics_contrib
#SBATCH -t 32:00:00 
#SBATCH -c 4
#SBATCH --mem=50GB
#SBATCH -o network_watcher.out

module load gcc/9.2.0
module load cuda/11.2
module load boost

increment=$1
final_iteration=$2
curr_iteration=$3
config_json=`realpath $4`
model=$5

# venv
segway_dir=/n/groups/htem/Segmentation/shared-nondev/segway2

#$?
# export PYTHONPATH=$PYTHONPATH:/n/groups/htem/Segmentation/tmn7/gunpowder-1.2.2-220114
# export PYTHONPATH=$PYTHONPATH:/n/groups/htem/users/jlr54/gunpowder_jax_triMod
# . /n/groups/htem/users/tmn7/envs/ubuntu18_python36_daisy1/bin/activate
# . /n/groups/htem/users/tmn7/envs/ubuntu20_python39_wk/bin/activate

# . /n/groups/htem/users/tmn7/envs/ubuntu18_python37/bin/activate
# . /n/groups/htem//users/mkn5/venv/daisy-2/bin/activate
if [ "$PYTHONPATH" != */n/groups/htem/Segmentation/shared-nondev* ]; then
    echo INFO: PYTHONPATH env does not have segway... adding it
    export PYTHONPATH=$PYTHONPATH:/n/groups/htem/Segmentation/shared-nondev
fi

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false

# Wait until network is running
while [ ! -f .running ] && [ ! -f .done ]
do
    sleep 1m
done

while [ $((curr_iteration-increment)) -ne $final_iteration ]
do

    # Check if checkpoint exists
    checkpoint_file="${model}_checkpoint_${curr_iteration}"
    resave_file="model_checkpoint_${curr_iteration}"          
    if [ -f "$checkpoint_file" ]; then
        mv $checkpoint_file $resave_file
    fi 
    if [ ! -f "$resave_file" ]; then
        echo "$checkpoint_file does not exist"
        sleep 5m
        continue
    fi
    if [ -f "$resave_file" ]; then
        echo "$checkpoint_file ready!"  

        # TODO: Determine if this is necessary
        # Remove existing zarr 
        if [ -d "${config_json/json/"zarr"}" ]; then
            echo "Removing zarr"
            rm -r "${config_json/json/"zarr"}"
        fi

        echo ==========Segment==========
        # TODO Check for waterz error
        python ${segway_dir}/tasks/segmentation/task_06a_extract_segments.py ${config_json}
        
        echo ==========Evaluate==========
        python /n/groups/htem/users/jlr54/raygun/Segmentation/rasterize_skeleton.py ${config_json} ${increment}
        
        # Increment
        curr_iteration=$((curr_iteration+increment))
        echo  Next iteration-- $curr_iteration
        sleep 1m
    fi
done

echo "all done"
