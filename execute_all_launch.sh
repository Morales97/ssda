#!/bin/bash


sbatch seed_launch.sh 1
sbatch seed_launch.sh 2
sbatch seed_launch.sh 3

#sbatch --dependency=afterok:$1 seed_launch.sh 1
#sbatch --dependency=afterok:$(($1+1)) seed_launch.sh 2
#sbatch --dependency=afterok:$(($1+1)) seed_launch.sh 3