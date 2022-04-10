#!/bin/bash
#
#SBATCH --job-name=two_jobs
#
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=30000
#SBATCH --time=03:00:00

CUDA_VISIBLE_DEVICES=0 python main_SSDA.py --project=GTA_CS_tiny_seeds --expt_name=COCO_bb --seed=2 --net=lraspp_mobilenet --pre_trained=True &
CUDA_VISIBLE_DEVICES=1 python main_SSDA.py --project=GTA_CS_tiny_seeds --expt_name=COCO_bb --seed=3 --net=lraspp_mobilenet --pre_trained=True

#CUDA_VISIBLE_DEVICES=0 python main_SSDA.py --project=GTA_CS_tiny_seeds --expt_name=CR_mask_pt_CS_sup_s2 --seed=2 --cr=prob_distr --custom_pretrain_path=model/pretrained/ckpt_mask_lraspp_CS_sup_s2.tar &
#CUDA_VISIBLE_DEVICES=1 python main_SSDA.py --project=GTA_CS_tiny_seeds --expt_name=CR_mask_pt_CS_sup_s3 --seed=3 --cr=prob_distr --custom_pretrain_path=model/pretrained/ckpt_mask_lraspp_CS_sup_s3.tar

#CUDA_VISIBLE_DEVICES=0 python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=CR_pd_IN_bb_s3 --seed=3 --cr=prob_distr &#--custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar &
#CUDA_VISIBLE_DEVICES=1 python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=CR_mask_pt_CS_s1 --seed=1 --cr=prob_distr --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar


#CUDA_VISIBLE_DEVICES=0 python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_CS_s4 --seed=4 --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar &
#CUDA_VISIBLE_DEVICES=1 python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_CS_s5 --seed=5 --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar


#wait