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

python main_SSDA.py --seed=2 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_no_blur --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --aug_level=3  #--custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_no_neg.tar
#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 #--custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_400.tar

#python main_FS.py --project=GTA_to_CS_tiny --expt_name=FS_IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 
#python main_FS.py --project=GTA_to_CS_small --size=small --expt_name=FS_IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 
#python main_FS.py --save_model=True --save_interval=5000 --steps=25000 --lr=0.001 --project=GTA_to_CS_tiny --expt_name=FS_25k_lr3 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 #--custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_400.tar

# DLv2
#python main_FS.py --seed=1 --save_interval=10000 --steps=100000 --lr=0.001 --lr_decay=poly --project=GTA_to_CS_tiny --expt_name=dlv2_FS_100k --net=deeplabv2_rn101 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 
#python main_FS.py --seed=1 --save_interval=50000 --steps=100000 --lr=0.001 --lr_decay=poly --project=GTA_to_CS_tiny --expt_name=dlv3_FS_100k --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 
