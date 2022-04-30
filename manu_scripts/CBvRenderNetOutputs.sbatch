#!/bin/bash

#SBATCH -p gpu_connectomics
#SBATCH --gres=gpu:1
#SBATCH --account=connectomics_contrib
#SBATCH -t 3-00:00
#SBATCH -c 16
#SBATCH --mem=50GB
#SBATCH -o /n/groups/htem/users/jlr54/logs/CBvRenderNetOutputs.out

module load gcc/6.2.0
module load cuda/10.1

python /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_train.py false 1024 350000 345000 340000 335000 330000