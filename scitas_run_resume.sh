#!/bin/bash
#
#SBATCH --job-name=resume2
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

python main_SSDA.py --seed=1 --steps=500 --steps_job=250 --final_job=True --project=GTA_CS_rn50_tiny --expt_name=test_launch --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --resume=expts/tmp_last/checkpoint_test_launch_1.pth.tar
