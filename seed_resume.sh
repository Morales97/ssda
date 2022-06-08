#!/bin/bash
#
#SBATCH --job-name=full500p2
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=full_500 --target_samples=500 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_500_$1.pth.tar
