#!/bin/bash
#
#SBATCH --job-name=alos2p2

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

#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=abl_tau0.5_p2 --cr=ce_th --tau=0.5 --pixel_contrast=True --resume=expts/tmp_last/checkpoint_abl_tau0.5_3.pth.tar
python main_SSDA.py --seed=2 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=abl_alo_p2 --cr=ce --alonso_contrast=full --resume=expts/tmp_last/checkpoint_abl_alo_2.pth.tar
#python main_SSDA_EMA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=full_rampupFIX_p2 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_rampupFIX_3.pth.tar
#python main_SSDA_EMA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_EMA_rampup_p2 --cr=ce --pixel_contrast=True --cutmix_cr=False --aug_level=4 --resume=expts/tmp_last/checkpoint_CE_EMA_rampup_3.pth.tar
#python main_SSDA_EMA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_EMA_rampupFIX_p2 --cr=ce --pixel_contrast=True --cutmix_cr=False --aug_level=4 --resume=expts/tmp_last/checkpoint_CE_EMA_rampupFIX_3.pth.tar
#python main_SSDA_EMA.py --seed=1 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=full --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_1.pth.tar

# 2nd round
#python main_SSDA.py --seed=1 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=full_r2_prevEMA_p2_rm --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_r2_prevEMA_1.pth.tar
#python main_SSDA.py --seed=2 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=full_r3_p2 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_r3_2.pth.tar
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=full_r2_p2_PL --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_r2_3.pth.tar --pseudolabel_folder=full_s3

# 3rd round
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=KL_pc_cw_r3_noPL --cr=kl --pixel_contrast=True --resume=expts/tmp_last/checkpoint_KL_pc_cw_r3_20k_3.pth.tar #--pseudolabel_folder=KL_pc_r23_test

# FULL, bs=1
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_full --batch_size_s=1 --batch_size_tl=1 --batch_size_tu=1 --cr=ce --pixel_contrast=True --pc_memory=True --alonso_contrast=full --resume=expts/tmp_last/checkpoint_CE_full_3.pth.tar

# UDA
#python main_UDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=UDA_CE_p2 --cr=ce --resume=expts/tmp_last/checkpoint_UDA_CE_3.pth.tar
#python main_SSDA_EMA.py --seed=3 --target_samples=0 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=UDA_CE_r2_p2 --cr=ce --pseudolabel_folder=UDA_CE_s1 --resume=expts/tmp_last/checkpoint_UDA_CE_r2_3.pth.tar

# SSL
#python main_SemiSup.py --seed=1 --steps=30000 --save_interval=30000 --steps_job=10000 --project=GTA_to_CS_small --expt_name=SemiSup_r2_p2 --cr=ce --pixel_contrast=True --pc_memory=True --resume=expts/tmp_last/checkpoint_SemiSup_r2_1.pth.tar
