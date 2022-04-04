#!/bin/bash
#
#SBATCH --job-name=test_simba
#
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=30000
#SBATCH --time=01:00:00

python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=expt_1 --net=lraspp_mobilenet --target_samples=100 --cr=one_hot --lmbda=5 &
python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=expt_2 --net=lraspp_mobilenet --target_samples=100 --cr=one_hot --lmbda=5