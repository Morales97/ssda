#!/bin/bash
#
#SBATCH --job-name=eval500
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
#python generate_pseudolabels.py --seed=$1 --expt_name=full_500_r2 --target_samples=500 --resume=expts/tmp_last/checkpoint_full_500_r2_$1.pth.tar
#python generate_pseudolabels.py --seed=$1 --expt_name=SS_372_r2 --target_samples=372 --resume=expts/tmp_last/checkpoint_SemiSup_372_r2_$1.pth.tar

# SS
#python generate_pseudolabels.py --seed=3 --expt_name=SSnomem_r2 --resume=expts/tmp_last/checkpoint_SemiSup_nomem_p2_3.pth.tar

# UDA
#python generate_pseudolabels.py --target_samples=0 --expt_name=UDA_CE --resume=expts/tmp_last/checkpoint_UDA_CE_3.pth.tar

# evaluate
python evaluate.py 

# evaluate for SSL
#python evaluate.py --pc_memory=True