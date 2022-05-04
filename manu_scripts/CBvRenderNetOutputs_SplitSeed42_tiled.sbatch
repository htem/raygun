#!/bin/bash

#SBATCH -p gpu_connectomics
#SBATCH --gres=gpu:1
#SBATCH --account=connectomics_contrib
#SBATCH -t 5-00:00 
#SBATCH -c 16
#SBATCH --mem=50GB
#SBATCH -o /n/groups/htem/users/jlr54/logs/CBvRenderNetOutputs_SplitSeed42_tiled.out

module load gcc/6.2.0
module load cuda/10.1

comm -23 --nocheck-order /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42_tiled.queue /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42_tiled.done > /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42_tiled.todo

while read p; do
    python /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_train.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5 volumes/interpolated_90nm_aligned false 200 $p
    echo $p >> /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42_tiled.done
done </n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42_tiled.todo
