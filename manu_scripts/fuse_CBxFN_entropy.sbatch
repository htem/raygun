#!/bin/bash

#SBATCH -p long
#SBATCH --account=htem
#SBATCH -t 30-00:00 
#SBATCH -c 34
#SBATCH --mem=816GB
#SBATCH -o /n/groups/htem/users/jlr54/logs/fuse_CBxFN_entropy.out

# export DISPLAY=:1
# Xvfb $DISPLAY -auth /dev/null &
# (
# the '(' starts a new sub shell. In this sub shell we start the worker processes:

# ~/fiji -macro /n/groups/htem/users/jlr54/raygun/manu_scripts/fuse_CBxFN_entropy.ijm # running the actual ijm script
/n/groups/htem/users/jlr54/Fiji/fiji --ij2 --headless --run /n/groups/htem/users/jlr54/raygun/manu_scripts/fuse_CBxFN_entropy.ijm # running the actual ijm script

# wait # waits until all 'program' processes are finished
# # this wait sees only the 'program' processes, not the Xvfb process
# )