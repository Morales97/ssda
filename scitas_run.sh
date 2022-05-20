#!/bin/bash
#
#SBATCH --job-name=sscepc
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00


#python main_SSDA_EMA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_EMA_rampupFIX --cr=ce --pixel_contrast=True --aug_level=4 --cutmix_cr=False
#python main_SSDA_EMA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=full_rampupFIX --cr=ce --pixel_contrast=True 
#python main_SSDA_EMA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=MSE --cr=mse --pixel_contrast=True 
#python main_SSDA_NOema.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_noEMA --cr=ce --pixel_contrast=True --aug_level=4 --cutmix_cr=False
#python main_SSDA_NOema.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=full_noEMA --cr=ce --pixel_contrast=True 

# SIZE TINY
#python main_SSDA.py --seed=1 --lr=0.01 --size=tiny --save_model=False --project=GTA_CS_rn50_tiny --expt_name=KL_CMcr_no_blur --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --class_weight=False --aug_level=3 --cutmix_cr=True #--aug_level=5 
#python main_SSDA_EMA.py --seed=1 --lr=0.01 --size=tiny --save_model=False --project=GTA_CS_rn50_tiny --expt_name=CE_EMA_new --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=ce 

# FULL, bs=1
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=CE_full --batch_size_s=1 --batch_size_tl=1 --batch_size_tu=1 --cr=ce --pixel_contrast=True --pc_memory=True --alonso_contrast=full 

# next round of ST
#python main_SemiSupNEW.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=KL_pc_cw_PL_noS --cr=kl --pixel_contrast=True --pseudolabel_folder=KL_pc_40k3_test
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=KL_r2_noeval --cr=kl --pixel_contrast=True --pseudolabel_folder=KL_40k_no_eval3 --aug_level=4 --cutmix_cr=False
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=KL_pc_cw_r3 --cr=kl --pixel_contrast=True --pseudolabel_folder=KL_pc_r23_test
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=KL_pc_cw_r4 --cr=kl --pixel_contrast=True --pseudolabel_folder=KL_r3_noPL3
#python main_SSDA.py --seed=3 --steps=40000 --save_interval=40000 --steps_job=20000 --project=GTA_to_CS_small --expt_name=KL_pc_cw_r2_teacher2 --cr=kl --pixel_contrast=True --pseudolabel_folder=KL_pc_40k3_test --teacher=model/pretrained/model_40k_KL_pc.tar


# UDA
#python main_UDA.py --seed=3 --steps=40000 --save_interval=40000 --project=GTA_to_CS_small --expt_name=UDA_CE --cr=ce 

# SSL
python main_SemiSup.py --seed=3 --steps=30000 --save_interval=30000 --project=GTA_to_CS_small --expt_name=SS_CE_pc --cr=ce --pixel_contrast=True