#!/bin/bash
#
#SBATCH --job-name=seg_test
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=30000
#SBATCH --time=12:00:00


python main.py --project=GTA_to_CS_tiny --expt_name=denseCL_fcn_rn50_FS_CS --net=denseCL_fcn_rn50 --steps=5000

# IN backbone
#python main_S_and_T_2.py --project=GTA_to_CS_tiny --expt_name=dummy2_100_same_size --net=lraspp_mobilenet --target_samples=100 --batch_size=8

# Rotation pretrained backbone
#python main_S_and_T_2.py --project=GTA_to_CS_tiny --expt_name=dummy2_100_same_size_rot_pt --net=lraspp_mobilenet --target_samples=100 --batch_size=8 --custom_pretrain_path=ckpt_rot_10k.tar

# Rotation
# python pretrain/rotation.py --batch_size=4 --lr=0.001 --project=rotation --expt_name=longer_run_lr_x10 --save_dir=pretrain/expts_rot/tmp_last --save_interval=2000 --steps=10001
