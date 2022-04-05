#!/bin/bash
#
#SBATCH --job-name=test_simba
#
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=30000
#SBATCH --time=03:00:00

CUDA_VISIBLE_DEVICES=0 python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_CS_s1 --net=lraspp_mobilenet --target_samples=100 --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar &
CUDA_VISIBLE_DEVICES=1 python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_CS_s2 --net=lraspp_mobilenet --target_samples=100 --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar

#wait