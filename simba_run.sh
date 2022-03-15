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


# python main.py --expt_name=lraspp_mnv3 --net=lraspp_mobilenet --pre_trained=False  --pre_trained_backbone=True --steps=5000
python main_S_and_T_2.py --project=GTA_to_CS_tiny --expt_name=dummy2_S_and_T_100 --net=lraspp_mobilenet --target_samples=100

