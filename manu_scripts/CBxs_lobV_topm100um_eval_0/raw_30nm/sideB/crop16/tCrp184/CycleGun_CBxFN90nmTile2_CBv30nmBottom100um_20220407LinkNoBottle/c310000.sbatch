#!/bin/bash 
#SBATCH -p gpu_connectomics 
#SBATCH --gres=gpu:1 
#SBATCH --account=connectomics_contrib 
#SBATCH -t 12:00:00  
#SBATCH -c 11 
#SBATCH --mem=50GB 
#SBATCH -o c310000.out 
module load gcc/9.2.0 
module load cuda/11.2 
python /n/groups/htem/users/jlr54/raygun/manu_scripts/render.py /n/groups/htem/users/jlr54/raygun/manu_scripts/CBxs_lobV_topm100um_eval_0/raw_30nm/sideB/crop16/tCrp184/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle/c310000.json 
