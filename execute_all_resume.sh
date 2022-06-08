#!/bin/bash

sbatch --dependency=afterok:$1 seed_resume.sh 1
sbatch --dependency=afterok:$1+1 seed_resume.sh 2
sbatch seed_resume.sh 3
