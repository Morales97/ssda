#!/bin/bash
#
#SBATCH --job-name=genPL
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

python generate_pseudolabels.py --seed=3 --expt_name=KL_40k_no_eval --resume=model/pretrained/model_40k_KL_pc.tar