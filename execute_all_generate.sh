#!/bin/bash


sbatch --dependency=afterok:$1 generate_pl.sh
sbatch --dependency=afterok:$(($1+1)) generate_pl.sh
sbatch --dependency=afterok:$(($1+2)) generate_pl.sh

