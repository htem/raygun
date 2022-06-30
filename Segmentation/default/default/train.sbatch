#!/bin/bash

#SBATCH -p gpu_connectomics
#SBATCH --gres=gpu:1
#SBATCH --account=connectomics_contrib
#SBATCH -t 32:00:00 
#SBATCH -c 6
#SBATCH --mem=50GB
#SBATCH -o train.out

module load gcc/9.2.0
module load cuda/11.2
export CUDA_VISIBLE_DEVICES=0

echo Started `date` > .running
python train.py train_kwargs.json
echo Finished `date` > .done
rm .running
