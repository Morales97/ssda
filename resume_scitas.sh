#!/bin/bash
#
#SBATCH --job-name=cealob
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00

#python main_SSDA.py --seed=1 --steps=200 --steps_job=100 --val_interval=100 --project=GTA_CS_rn50_tiny --expt_name=test_launch2 --net=deeplabv3_rn50 --batch_size_s=4 --batch_size_tl=4 --batch_size_tu=4 --cr=kl --resume=expts/tmp_last/checkpoint_test_launch2_1.pth.tar
python main_SSDA.py --seed=3 --lr=0.001 --lr_decay=det --steps=40000 --save_interval=40000 --project=GTA_to_CS_small --size=small --expt_name=CE_alo_base --net=deeplabv2_rn101 --batch_size_s=2 --batch_size_tl=2 --batch_size_tu=2 --cr=ce --alonso_contrast=base --resume=expts/tmp_last/checkpoint_CE_alo_base_3.pth.tar
