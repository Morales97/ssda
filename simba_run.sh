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
#SBATCH --time=08:00:00


python main.py --expt_name=fcn_rn50 --net=resnet50_fcn --pre_trained=False  --pre_trained_backbone=True --lr=0.01 --steps=5000

