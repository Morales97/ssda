#!/bin/bash
#
#SBATCH --job-name=deeplab
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00


python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=PC_CR_mask_CS --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 --custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS.tar --cr=prob_distr --pixel_contrast=True

#python main_FS.py --project=GTA_to_CS_tiny --expt_name=FS_IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 
#python main_FS.py --project=GTA_to_CS_small --size=small --expt_name=FS_IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 

