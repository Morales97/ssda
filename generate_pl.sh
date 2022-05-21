#!/bin/bash
#
#SBATCH --job-name=genPLs2
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

# SSDA
python generate_pseudolabels.py --seed=2 --expt_name=full --resume=expts/tmp_last/checkpoint_full_2.pth.tar

# SS
#python generate_pseudolabels.py --seed=3 --expt_name=SS_CE_pc_mem_r2 --resume=expts/tmp_last/checkpoint_SS_CE_pc_mem_r2_3.pth.tar --pc_memory=True

# UDA
#python generate_pseudolabels.py --target_samples=0 --expt_name=UDA_CE --resume=expts/tmp_last/checkpoint_UDA_CE_3.pth.tar

# evaluate
#python evaluate.py --net=deeplabv2_rn101 --alonso_contrast=full --pc_memory=True --resume=expts/tmp_last/checkpoint_KL_pc_cw_r3_noPL_3.pth.tar