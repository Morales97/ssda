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

python generate_pseudolabels.py --seed=3 --size=small --expt_name=KL_pc_r2 --net=deeplabv2_rn101 --resume=expts/tmp_last/checkpoint_KL_pc_cw_PL_3.pth.tar