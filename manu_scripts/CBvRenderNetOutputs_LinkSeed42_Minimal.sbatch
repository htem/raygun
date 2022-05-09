#!/bin/bash

#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --account=htem
#SBATCH --requeue
#SBATCH -t 6:00:00 
#SBATCH -c 16
#SBATCH --mem=250GB
#SBATCH -o /n/groups/htem/users/jlr54/logs/CBvRenderNetOutputs_LinkSeed42_Minimal.out

comm -23 --nocheck-order /n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_LinkSeed42.queue /n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_LinkSeed42.done > /n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_LinkSeed42.todo

while read p; do
    python /n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_Minimal.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/raygun/CycleGAN/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_train.py /n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvTopGT/CBxs_lobV_topm100um_eval_0.n5 volumes/interpolated_90nm_aligned netG1 362 $p
    echo $p >> /n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_LinkSeed42.done
done </n/groups/htem/users/jlr54/queues/CBvRenderNetOutputs_LinkSeed42.todo
