#!/bin/bash


sbatch --dependency=afterok:$1 generate_pl.sh 1
sbatch --dependency=afterok:$(($1+1)) generate_pl.sh 2
sbatch --dependency=afterok:$(($1+2)) generate_pl.sh 3

