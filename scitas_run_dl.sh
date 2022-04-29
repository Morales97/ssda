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

python main_SSDA.py --seed=2 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KLE_alonso_unl --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --alonso_contrast=True --cr=kl
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=mask_CS_GTA --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --custom_pretrain_path=model/pretrained/ckpt_mask_v3_CS_GTA_400.tar
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=LAB_color --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --lab_color=True
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KLE_LAB_color_norm --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --lab_color=True
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KLE_GTA_cycada --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=pixpro --net=deeplabv3_rn50_pixpro --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4

#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KL_EMA_PC2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --warmup_steps=2000 --resume=model/pretrained/ckpt_KLE_PC2_10k.tar
#python main_SemiSup.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=SS_KLE --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl

