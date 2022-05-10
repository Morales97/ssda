#!/bin/bash
#
#SBATCH --job-name=launchpy
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=500
#SBATCH --time=00:01:00

python launch_slurm.py
