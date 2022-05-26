#!/bin/bash
#
#SBATCH --job-name=fulls1
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00


#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=abl_tau0.5 --cr=ce_th --tau=0.5 --pixel_contrast=True 
python main_SSDA_noEMA.py --save_model=False --seed=1 --steps=40000 --save_interval=40000 --project=clean_runs --expt_name=abl_d100 
python main_FS.py --save_model=False --seed=1 --steps=40000 --project=clean_runs --expt_name=FS
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=MSE --cr=mse --pixel_contrast=True 

# SIZE TINY
#python main_SSDA.py --seed=1 --lr=0.01 --size=tiny --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_CMcr_no_blur --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --class_weight=False --aug_level=3 --cutmix_cr=True #--aug_level=5 
#python main_SSDA_EMA.py --seed=1 --lr=0.01 --size=tiny --save_model=False --project=GTA_CS_rn50_tiny --expt_name=CE_EMA_new --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=ce 

# FULL, bs=1
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_full --batch_size_s=1 --batch_size_tl=1 --batch_size_tu=1 --cr=ce --pixel_contrast=True --pc_memory=True --alonso_contrast=full 

# next round of ST
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=clean_runs --expt_name=full_r2_prevEMA --cr=ce --pixel_contrast=True --pseudolabel_folder=full_s3 --prev_teacher=expts/tmp_last/checkpoint_full_p2_3.pth.tar
#python main_SSDA_EMA.py --seed=2 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=full_r3 --cr=ce --pixel_contrast=True --pseudolabel_folder=full_r2_s2

# UDA
#python main_UDA.py --seed=3 --steps=40000 --save_interval=40000 --project=GTA_to_CS_small --expt_name=UDA_CE --cr=ce 
# UDA ST
#python main_SSDA_EMA.py --seed=3 --target_samples=0 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=UDA_CE_r2 --cr=ce --pseudolabel_folder=UDA_CE_s1

# SSL
#python main_SemiSup.py --seed=3 --steps=30000 --save_interval=30000 --project=GTA_to_CS_small --expt_name=SS_CE_pc_mem --cr=ce --pixel_contrast=True --pc_memory=True
#python main_SemiSup.py --seed=2 --steps=30000 --save_interval=30000 --project=clean_runs --expt_name=SemiSup --cr=ce --pixel_contrast=True --pc_memory=True
#python main_SemiSup.py --seed=1 --steps=30000 --save_interval=30000 --project=clean_runs --expt_name=SemiSup_r2 --cr=ce --pixel_contrast=True --pc_memory=True --pseudolabel_folder=SS_s1
#python main_SemiSup.py --seed=3 --steps=30000 --save_interval=30000 --project=GTA_to_CS_small --expt_name=SS_CE_pc_mem_r3 --cr=ce --pixel_contrast=True --pc_memory=True --pseudolabel_folder=SS_CE_pc_mem_r2_s3
#python main_SemiSup.py --seed=1 --steps=30000 --save_interval=30000 --project=clean_runs --expt_name=SemiSup_r3 --cr=ce --pixel_contrast=True --pc_memory=True --dropPL_step=20000 --pseudolabel_folder=SS_r2_s1
#python main_SemiSup.py --seed=3 --steps=30000 --save_interval=30000 --project=GTA_to_CS_small --expt_name=SS_CE_pc_mem_r2cont_noPL --cr=ce --pixel_contrast=True --pc_memory=True --custom_pretrain=expts/tmp_last/checkpoint_SS_CE_pc_mem_r2_3.pth.tar