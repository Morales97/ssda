#!/bin/bash
#
#SBATCH --job-name=gens3
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
#python generate_pseudolabels.py --seed=3 --expt_name=abl_noPCmix --resume=expts/tmp_last/checkpoint_abl_noPCmix_p2_3.pth.tar

# SS
python generate_pseudolabels.py --seed=3 --expt_name=SSnew500 --resume=expts/tmp_last/checkpoint_SemiSupNEW_500_3.pth.tar --pc_memory=True

# UDA
#python generate_pseudolabels.py --target_samples=0 --expt_name=UDA_CE --resume=expts/tmp_last/checkpoint_UDA_CE_3.pth.tar

# evaluate
#python evaluate.py 

# evaluate for SSL
#python evaluate.py --pc_memory=True