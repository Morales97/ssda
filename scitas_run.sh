#!/bin/bash
#
#SBATCH --job-name=gjs
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_pc_mem --net=deeplabv3_rn50_mem --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --pixel_contrast=True --pc_memory=True
#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=maskC_on_T --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --custom_pretrain=model/pretrained/ckpt_mask_dlrn50_CS_400.tar
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=alonso_full_tau0 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --alonso_contrast=full

python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=GJS_pc_cw --cr=gjs --n_augmentations=2 --pixel_contrast=True --class_weight=True
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=GJS_pc_cw --cr=gjs --n_augmentations=2 --pixel_contrast=True --class_weight=True
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_pc_cw_cutmixCR --cr=ce --pixel_contrast=True --cutmix_cr=True
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_pc_mem_cw --cr=ce --pixel_contrast=True --pc_memory=True --class_weight=True
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=KL_pc_cw_PL --cr=kl --pixel_contrast=True --pseudolabel_folder=KL_pc_40k3_test
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --project=GTA_to_CS_small --expt_name=KL_pc_cw_p2 --cr=kl --pixel_contrast=True --resume=expts/tmp_last/checkpoint_KL_pc_cw.pth.tar
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --expt_name=CE_full_bs1_p2 --batch_size_s=1 --batch_size_tl=1 --batch_size_tu=1 --cr=ce --pixel_contrast=True --alonso_contrast=full --resume=expts/tmp_last/checkpoint_CE_full_bs1.pth.tar
#python main_SSDA.py --save_model=False --seed=3 --steps=40000 --save_interval=5000 --project=GTA_to_CS_small --expt_name=KL_full_equip_v2 --cr=kl --pixel_contrast=True --cutmix_sup=True --cutmix_cr=True --resume=expts/tmp_last/
#python main_SemiSup.py --save_model=False --seed=1 --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --expt_name=SS_KLE_alo_tau0_cw --cr=kl --alonso_contrast=full #--pixel_contrast=True 

# SIZE TINY
#python main_SSDA.py --seed=1 --lr=0.01 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_pc_mem --size=tiny --net=deeplabv3_rn50_mem --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --pixel_contrast=True --pc_memory=True

# FULL, bs=1
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_full --batch_size_s=1 --batch_size_tl=1 --batch_size_tu=1 --cr=ce --pixel_contrast=True --pc_memory=True --alonso_contrast=full 

# next round of ST
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=KL_pc_cw_PL_2 --cr=kl --pixel_contrast=True --pseudolabel_folder=KL_pc_40k3_test
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=KL_pc_cw_r3 --cr=kl --pixel_contrast=True --pseudolabel_folder=KL_pc_r23_test

# UDA
#python main_UDA.py --seed=3 --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --expt_name=UDA_CE_alo --cr=ce --alonso_contrast=full