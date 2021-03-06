#!/bin/bash
#
#SBATCH --job-name=500
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

#python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=abl_100_no_mix_batch --cr=ce --pixel_contrast=True 
#python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=full_2975 --target_samples=2975 --cr=ce --pixel_contrast=True
#python main_SemiSup.py --seed=$1 --steps_job=20000 --project=clean_SSL --expt_name=SemiSup_744_r3 --target_samples=744 --cr=ce --pixel_contrast=True --pseudolabel_folder=SS_744_r2_s$1

#python main_SemiSup.py --seed=$1 --steps_job=20000 --project=clean_SSL --expt_name=SemiSup_2975 --target_samples=2975 --cr=ce --pixel_contrast=True
python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=full_500_r2_nodrop --target_samples=500 --cr=ce --pixel_contrast=True --pseudolabel_folder=full_500_s$1
#python main_SemiSup.py --seed=$1 --steps_job=20000 --project=clean_SSL --expt_name=SemiSup_744_r2_nodrop --target_samples=744 --cr=ce --pixel_contrast=True --pseudolabel_folder=SS_744_s$1

