#!/bin/bash
#
#SBATCH --job-name=100cscw
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=CE_1_rerun --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=ce #--cutmix_cr=True #--pixel_contrast=True --pc_mixed=True
#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=maskC_on_T --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --custom_pretrain=model/pretrained/ckpt_mask_dlrn50_CS_400.tar
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=alonso_full_tau0 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --alonso_contrast=full
#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=GJS_no_ema --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=gjs --n_augmentations=2
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_no_ema_no_grad --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --warmup_steps=5000
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_oh_0.1 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl_oh --lmbda=0.1
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=JS_dsbn_l1 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=js --cr_ema=False --dsbn=True
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=JS_oh --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=js_oh --lmbda=0.01

#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=dsbn_l1_alonso_tu_0.9 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --alonso_contrast=base --dsbn=True
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=alonso_FQ --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --alonso_contrast=feat_quality
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=CE_no_ema --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=ce --cr_ema=False #--custom_pretrainmodel/pretrained/ckpt_mask_dlrn50_CS_400.tar
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=LAB_color --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --lab_color=True
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KLE_GTA_cycada --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl

#python main_SSDA.py --save_model=False --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=5000 --project=GTA_to_CS_small --size=small --expt_name=KL_full_equip --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --pc_mixed=True --cutmix_cr=True --cutmix_sup=True --custom_pretrain=model/pretrained/ckpt_mask_v2.tar
python main_SSDA.py --save_model=False --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=5000 --project=GTA_to_CS_small --size=small --expt_name=d100_cw --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --class_weight=True
#python main_SemiSup.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --expt_name=100cs_cw --net=deeplabv2_rn101 --size=small --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --class_weight=True #--cr=kl --alonso_contrast=full --pixel_contrast=True --pc_mixed=True 
#python main_SemiSup.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=100_CS --net=deeplabv3_rn50 --batch_size_tl=4 --batch_size_tu=4 

