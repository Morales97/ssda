#!/bin/bash
#
#SBATCH --job-name=klcw2
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_cw2 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --class_weight=True
#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=maskC_on_T --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --custom_pretrain=model/pretrained/ckpt_mask_dlrn50_CS_400.tar
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=alonso_full_tau0 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --alonso_contrast=full
#python main_SSDA.py --seed=1 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=GJS_no_ema --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=gjs --n_augmentations=2
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_no_ema_no_grad --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --warmup_steps=5000
#python main_SSDA.py --seed=3 --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_oh_0.1 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl_oh --lmbda=0.1

#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --size=small --expt_name=CE_pc_mask --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=ce --pixel_contrast=True --pc_mixed=True --custom_pretrain=model/pretrained/ckpt_mask_v2.tar
#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --project=GTA_to_CS_small --size=small --expt_name=KL_pc_cw_p2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --pc_mixed=True --class_weight=True --resume=expts/tmp_last/checkpoint_KL_pc_cw.pth.tar
#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=CE_full_bs1_p2 --net=deeplabv2_rn101 --batch_size_s=1 --batch_size_tl=1 --batch_size_tu=1 --cr=ce --pixel_contrast=True --pc_mixed=True --alonso_contrast=full --resume=expts/tmp_last/checkpoint_CE_full_bs1.pth.tar
#python main_SSDA.py --save_model=False --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=5000 --project=GTA_to_CS_small --size=small --expt_name=KL_full_equip_v2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --pc_mixed=True --cutmix_sup=True --cutmix_cr=True --resume=expts/tmp_last/
#python main_SemiSup.py --save_model=False --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=10000 --project=GTA_to_CS_small --expt_name=SS_KLE_alo_tau0_cw --net=deeplabv2_rn101 --size=small --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --class_weight=True --cr=kl --alonso_contrast=full #--pixel_contrast=True --pc_mixed=True 

# next round of ST
#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=10000 --save_interval=10000 --project=GTA_to_CS_small --size=small --expt_name=KL_pc_round2_noS --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --pc_mixed=True --warmup_steps=0 --pseudolabel_folder=KL_pc_40k3_test --round_start=model/pretrained/model_40k_KL_pc.tar
