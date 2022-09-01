#!/bin/bash

#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --account=htem
#SBATCH --requeue
#SBATCH -t 6:00:00 
#SBATCH -c 16
#SBATCH --mem=25GB
#SBATCH -o /n/groups/htem/users/jlr54/logs/CBvRenderNetOutputs_SplitSeed13_Minimal.out

comm -23 --nocheck-order /n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_SplitSeed13.queue /n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_SplitSeed13.done > /n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_SplitSeed13.todo

while read p; do
    python /n/groups/htem/users/jlr54/raygun/manu_scripts/CBvRenderNetOutputs_Minimal.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407-2SplitNoBottle_train.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5 volumes/raw_30nm netG2 362 $p
    echo $p >> /n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_SplitSeed13.done
done </n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_SplitSeed13.todo
