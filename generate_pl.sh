#!/bin/bash
#
#SBATCH --job-name=genPL
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
#python generate_pseudolabels.py --seed=3 --expt_name=UDA_CE --resume=expts/tmp_last/checkpoint_UDA_CE_3.pth.tar

# UDA
python generate_pseudolabels.py --target_samples=0 --expt_name=UDA_CE --resume=expts/tmp_last/checkpoint_UDA_CE_3.pth.tar

# evaluate
#python evaluate.py --net=deeplabv2_rn101 --alonso_contrast=full --pc_memory=True --resume=expts/tmp_last/checkpoint_KL_pc_cw_r3_noPL_3.pth.tar