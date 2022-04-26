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
#SBATCH --time=04:00:00

python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=maskC_no_neg --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_no_neg.tar
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KL_EMA_PC_mask --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --custom_pretrain_path=model/pretrained/ckpt_mask_v2.tar
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=pixpro --net=deeplabv3_rn50_pixpro --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4


#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=IN_bb_mask_random --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --custom_pretrain_path=model/pretrained/ckpt_mask_dlrn50_CS_random_neg.tar
#python main_SSDA.py --seed=1 --lr=0.01 --lr_decay=poly --steps=30000 --save_interval=15000 --project=GTA_to_CS_small --size=small --expt_name=KL --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KL_EMA_sde_config_v2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl 
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KL_PC_sde_config_v2_only_T --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --warmup_steps=2000
#python main_SSDA.py --seed=1 --lr=0.01 --lr_decay=poly --steps=30000 --save_interval=15000 --project=GTA_to_CS_small --size=small --expt_name=KL --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --resume=model/pretrained/ckpt_v3_KL_15k.tar
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KL_v3_part3 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --resume=model/pretrained/ckpt_KL_v2_part2_30k.tar

#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=maskC --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --custom_pretrain_path=model/pretrained/ckpt_mask_v2.tar

#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=dummy100_v2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 #--resume=model/pretrained/ckpt_v2_KL_PC_10k.tar
#python main_SSDA.py --seed=1 --lr=0.01 --lr_decay=poly --steps=30000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=dummy_100_dsbn --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --dsbn=True

# DLv2
#python main_FS.py --seed=1 --save_interval=10000 --steps=100000 --lr=0.001 --lr_decay=poly --project=GTA_to_CS_tiny --expt_name=dlv2_FS_100k --net=deeplabv2_rn101 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 
#python main_FS.py --seed=1 --save_interval=25000 --steps=100000 --lr=0.001 --lr_decay=poly --project=GTA_to_CS_tiny --expt_name=dlv3_FS_100k_lr3 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 
#python main_SSDA.py --seed=1 --save_interval=25000 --steps=100000 --lr=0.01 --lr_decay=poly --project=GTA_to_CS_tiny --expt_name=dlv3_KL_100k_lr2 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl

#python main_SSDA.py --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --dsbn=True --log_interval=2
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=dsbn_IN_bb --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --dsbn=True
