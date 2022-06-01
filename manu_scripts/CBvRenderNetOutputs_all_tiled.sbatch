#!/bin/bash

#SBATCH -p gpu_connectomics
#SBATCH --gres=gpu:1
#SBATCH --account=connectomics_contrib
#SBATCH -t 5-00:00 
#SBATCH -c 30
#SBATCH --mem=50GB
#SBATCH -o /n/groups/htem/users/jlr54/logs/CBvRenderNetOutputs_all_tiled.out

module load gcc/6.2.0
module load cuda/10.1

python /n/groups/htem/users/jlr54/raygun/manu_scripts/render_all.py
