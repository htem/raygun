#!/bin/bash
#SBATCH -t 11:40:00
#SBATCH -p short
#SBATCH -c 1
#SBATCH --mem=2GB
#SBATCH -o .logs/RechunkTask_188779631922283499005426520668276887125_%j.out
#SBATCH -e .logs/RechunkTask_188779631922283499005426520668276887125_%j.err

python /n/groups/htem/users/jlr54/raygun/segway/tasks/rechunk/task_rechunk.py run_worker .run_configs/RechunkTask_188779631922283499005426520668276887125.config