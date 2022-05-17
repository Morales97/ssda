#!/bin/bash
#
#SBATCH --job-name=klr3p2
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

#python main_SSDA.py --seed=1 --steps=200 --steps_job=100 --val_interval=100 --project=GTA_CS_rn50_tiny --expt_name=test_launch2 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --resume=expts/tmp_last/checkpoint_test_launch2_1.pth.tar

#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --size=small --expt_name=CE_pc_cw_cutmixCR_p2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=ce --pixel_contrast=True --pc_mixed=True --class_weight=True --cutmix_cr=True --resume=expts/tmp_last/checkpoint_CE_pc_cw_cutmixCR_3.pth.tar
#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --size=small --expt_name=KL_pc_cw_PL_p2_noPL --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --pc_mixed=True --class_weight=True --resume=expts/tmp_last/checkpoint_KL_pc_cw_PL_20k_3.pth.tar --pseudolabel_folder=KL_pc_40k3_test
#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --size=small --expt_name=CE_pc_mem_cw_p2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=ce --pixel_contrast=True --pc_mixed=True --pc_memory=True --class_weight=True --resume=expts/tmp_last/checkpoint_CE_pc_mem_cw_3.pth.tar
#python main_SSDA.py --seed=1 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --size=small --expt_name=CE_pc_cw_p2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=ce --pixel_contrast=True --pc_mixed=True --class_weight=True --resume=expts/tmp_last/checkpoint_CE_pc_cw_1.pth.tar
#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --size=small --expt_name=CE_pc_mask --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=ce --pixel_contrast=True --pc_mixed=True --resume=expts/tmp_last/checkpoint_CE_pc_mask_3.pth.tar

# 2nd round
#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --size=small --expt_name=KL_pc_cw_PL_2_p2 --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --pc_mixed=True --class_weight=True --resume=expts/tmp_last/checkpoint_KL_pc_cw_PL_2_3.pth.tar --pseudolabel_folder=KL_pc_40k3_test

# 3rd round
python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --size=small --expt_name=KL_pc_cw_r3_noPL --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=kl --pixel_contrast=True --pc_mixed=True --class_weight=True --resume=expts/tmp_last/checkpoint_KL_pc_cw_r3_20k_3.pth.tar #--pseudolabel_folder=KL_pc_r23_test


# FULL, bs=1
#python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --size=small --expt_name=CE_full --net=deeplabv2_rn101 --batch_size_s=1 --batch_size_tl=1 --batch_size_tu=1 --cr=ce --pixel_contrast=True --pc_mixed=True --pc_memory=True --alonso_contrast=full --class_weight=True --resume=expts/tmp_last/checkpoint_CE_full_3.pth.tar
