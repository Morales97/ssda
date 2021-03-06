#!/bin/bash
#
#SBATCH --job-name=200p2
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

#python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=abl_100_no_mix_batch --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_abl_100_no_mix_batch_$1.pth.tar
#python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=full_500_r3 --target_samples=500 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_500_r3_$1.pth.tar
#python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=full_2975 --target_samples=2975 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_full_2975_$1.pth.tar
#python main_SemiSup.py --seed=$1 --steps_job=20000 --project=clean_SSL --expt_name=SemiSup_744_r3 --target_samples=744 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_SemiSup_744_r3_$1.pth.tar

#python main_SemiSup.py --seed=$1 --steps_job=20000 --project=clean_SSL --expt_name=SemiSup_2975 --target_samples=2975 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_SemiSup_2975_$1.pth.tar
#python main_SemiSup.py --seed=$1 --steps_job=20000 --project=clean_SSL --expt_name=SemiSup_744_r2_nodrop --target_samples=744 --cr=ce --pixel_contrast=True --resume=expts/tmp_last/checkpoint_SemiSup_744_r2_nodrop_$1.pth.tar --pseudolabel_folder=SS_744_s$1
python main_SSDA.py --seed=$1 --steps_job=20000 --project=clean_runs --expt_name=full_200_r2_nodrop --target_samples=200 --cr=ce --pixel_contrast=True --pseudolabel_folder=full_200_s$1 --resume=expts/tmp_last/checkpoint_full_200_r2_nodrop_$1.pth.tar 
