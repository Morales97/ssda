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

#python main_SSDA.py --save_model=False --project=GTA_to_CS_tiny --expt_name=CR_IN_bb_bs2 --net=deeplabv3_rn50 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --seed=1 #--custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_400.tar
#python main_SSDA.py --save_model=False --project=GTA_to_CS_tiny --expt_name=IN_bb_lr_poly --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 #--custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_400.tar
python main_FS.py --save_model=True --save_interval=5000 --steps=25000 --project=GTA_to_CS_tiny --expt_name=FS_25k_lr2 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 #--custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_400.tar

#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=CR_JS_IN_bb_2_augs_detach --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 --cr=js --tau=0 --n_augmentations=2 
#--custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS.tar 

#python main_FS.py --project=GTA_to_CS_tiny --expt_name=FS_IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 
#python main_FS.py --project=GTA_to_CS_small --size=small --expt_name=FS_IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 

