#!/bin/bash
#
#SBATCH --job-name372r2p2
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

#python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=full_200_r2 --target_samples=200 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_200_r2_$1.pth.tar
#python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=full_2975 --target_samples=2975 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_2975_$1.pth.tar
python main_SemiSup.py --seed=$1 --steps_job=20000 --project=clean_SSL --expt_name=SemiSup_372_r2 --target_samples=372 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_SemiSup_372_r2_$1.pth.tar
