#!/bin/bash

#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --account=htem
#SBATCH -t 6:00:00 
#SBATCH --requeue
#SBATCH -c 16
#SBATCH --mem=50GB
#SBATCH -o /n/groups/htem/users/jlr54/logs/CBvRenderNetOutputs_SplitSeed42_Minimal.out

module load gcc/6.2.0
module load cuda/11.1  

comm -23 --nocheck-order /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42.queue /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42.done > /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42.todo

while read p; do
    python /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_Minimal.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_train.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5 volumes/interpolated_90nm_aligned netG1 312 $p
    echo $p >> /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42.done
done </n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_SplitSeed42.todo
