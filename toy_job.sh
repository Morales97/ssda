#!/bin/bash
#
#SBATCH --job-name=toy_job
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=100
#SBATCH --time=00:01:00

ECHO "toy job"
