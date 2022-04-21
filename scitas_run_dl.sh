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

#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_EMA_PC_mix_pixpro --net=deeplabv3_rn50_pixpro --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --pixel_contrast=True --pc_mixed=True #--custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_400.tar
#python main_SSDA.py --seed=1 --lr=0.01 --lr_decay=poly --steps=30000 --save_interval=15000 --project=GTA_to_CS_small --size=small --expt_name=KL --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KL_EMA_sde_config_v2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl 
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KL_PC_sde_config_v2_only_T --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --warmup_steps=2000
#python main_SSDA.py --seed=1 --lr=0.01 --lr_decay=poly --steps=30000 --save_interval=15000 --project=GTA_to_CS_small --size=small --expt_name=KL --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --resume=model/pretrained/ckpt_v3_KL_15k.tar

python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=IN_bb --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 #--resume=model/pretrained/ckpt_v2_KL_PC_10k.tar

#python main_FS.py --project=GTA_to_CS_tiny --expt_name=FS_IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 
#python main_FS.py --lr=0.01 --lr_decay=poly --steps=30000 --save_interval=15000 --project=GTA_to_CS_small --size=small --expt_name=FS_clip --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1
#python main_FS.py --lr=0.001 --lr_decay=poly --steps=30000 --save_interval=15000 --project=GTA_to_CS_small --size=small --expt_name=FS_lr3_bs2 --net=deeplabv3_rn50 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --seed=1 
#python main_FS.py --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=FS_sde_config_v2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --seed=1 
#python main_FS.py --save_model=True --save_interval=5000 --steps=25000 --lr=0.001 --project=GTA_to_CS_tiny --expt_name=FS_25k_lr3 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --seed=1 #--custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_400.tar

# DLv2
#python main_FS.py --seed=1 --save_interval=10000 --steps=100000 --lr=0.001 --lr_decay=poly --project=GTA_to_CS_tiny --expt_name=dlv2_FS_100k --net=deeplabv2_rn101 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 
#python main_FS.py --seed=1 --save_interval=25000 --steps=100000 --lr=0.001 --lr_decay=poly --project=GTA_to_CS_tiny --expt_name=dlv3_FS_100k_lr3 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 
#python main_SSDA.py --seed=1 --save_interval=25000 --steps=100000 --lr=0.01 --lr_decay=poly --project=GTA_to_CS_tiny --expt_name=dlv3_KL_100k_lr2 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl

#python main_SSDA.py --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --dsbn=True --log_interval=2
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=dsbn_IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --dsbn=True
