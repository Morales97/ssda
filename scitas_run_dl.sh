#!/bin/bash
#
#SBATCH --job-name=ce_alo
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=04:00:00

python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=CE_tau0_alonso_full --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --alonso_contrast=full --cr=prob_distr --tau=0 
#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=GJS_no_ema --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=gjs --n_augmentations=2
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_no_ema_no_grad --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --warmup_steps=5000
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=CR_oh --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=one_hot
#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=JS_0.9 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=js_th 
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=JS_oh --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=js_oh --lmbda=0.01

#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=dsbn_l1_alonso_tu_0.9 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --alonso_contrast=base --dsbn=True
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=alonso_FQ --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --alonso_contrast=feat_quality
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=mask_CS_GTA --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --custom_pretrain=model/pretrained/ckpt_mask_dlrn50_CS_GTA_200.tar
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=LAB_color --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --lab_color=True
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KLE_GTA_cycada --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl

#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KL_EMA_PC2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --warmup_steps=2000 --resume=model/pretrained/ckpt_KLE_PC2_10k.tar
#python main_SemiSup.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=100_CS --net=deeplabv3_rn50 --batch_size_tl=4 --batch_size_tu=4 
#python main_SemiSup.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --expt_name=SS_KLE --net=deeplabv2_rn101 --size=small --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --warmup_steps=2000 

