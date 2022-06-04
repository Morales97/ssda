#!/bin/bash
#
#SBATCH --job-name=PCp2s3
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00


#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=noPCmix_mem --cr=ce --pixel_contrast=True --pc_memory=True --pc_mixed=False --alpha=0.99 --resume=expts/tmp_last/checkpoint_noPCmix_mem_3.pth.tar
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=full_new --cr=ce --pixel_contrast=True --alpha=0.99 --resume=expts/tmp_last/checkpoint_full_new_3.pth.tar

#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=abl_tau0.5_p2 --cr=ce_th --tau=0.5 --pixel_contrast=True --resume=expts/tmp_last/checkpoint_abl_tau0.5_3.pth.tar
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=full_newPC --cr=ce --pixel_contrast=True --pc_memory=True --resume=expts/tmp_last/checkpoint_full_newPC_3.pth.tar

# 2nd round
python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=abl_noPCmix_r3 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_abl_noPCmix_r3_3.pth.tar

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
#python main_SemiSup.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_SSL --expt_name=SemiSup_r2_40k_p2 --cr=ce --pixel_contrast=True --pc_memory=True --resume=expts/tmp_last/checkpoint_SemiSup_r2_40k_3.pth.tar
#python main_SemiSup.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_SSL --expt_name=SemiSupNEW --cr=ce --pixel_contrast=True --pc_memory=True --pc_ema=True --alpha=0.99 --resume=expts/tmp_last/checkpoint_SemiSupNEW_3.pth.tar
#python main_SemiSup.py --seed=1 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_SSL --expt_name=SemiSupNEW_r2 --cr=ce --pixel_contrast=True --pc_memory=True --pc_ema=True --alpha=0.99 --resume=expts/tmp_last/checkpoint_SemiSupNEW_r2_1.pth.tar
#python main_SemiSup.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_SSL --expt_name=SemiSup_nomem_r3 --cr=ce --pixel_contrast=True --pc_memory=False --resume=expts/tmp_last/checkpoint_SemiSup_nomem_r2_3.pth.tar


