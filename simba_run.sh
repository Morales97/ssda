#!/bin/bash
#
#SBATCH --job-name=seg_test
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=30000
#SBATCH --time=01:00:00


python main.py --expt_name=deeplabv3_rn50_no_pt_backbone --pre_trained=False  --net=deeplabv3 --pre_trained_backbone=False

