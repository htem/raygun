#!/bin/bash
#SBATCH -t 11:40:00
#SBATCH -p short
#SBATCH -c 1
#SBATCH --mem=2GB
#SBATCH -o .logs/RechunkTask_303445100715415620276211398463205460688_%j.out
#SBATCH -e .logs/RechunkTask_303445100715415620276211398463205460688_%j.err

python /n/groups/htem/users/jlr54/raygun/segway/tasks/rechunk/task_rechunk.py run_worker .run_configs/RechunkTask_303445100715415620276211398463205460688.config