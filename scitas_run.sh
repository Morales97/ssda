#!/bin/bash
#
#SBATCH --job-name=seg_test
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=03:00:00

# ----- Fully supervised CS -----
#python main.py --project=GTA_to_CS_tiny --expt_name=deeplab_rn50_FS_CS_pt_mask --net=deeplabv3_mask_pt --steps=5000
#python main.py --project=GTA_to_CS_tiny --expt_name=only_100_CS --net=lraspp_mobilenet --steps=5000 --target_samples=100 

# ----- Domain Adaptation -----
# -- DeepLab --
# python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_PASCAL_lr3 --net=deeplabv3_mask_pt --target_samples=100 --custom_pretrain_path=model/pretrained/VOCSegmentation_supervised_saliency_model.pth.tar 
# -- LR-ASPP --
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_CS_100sup_top5_GTAcrops --net=lraspp_mobilenet --target_samples=100 --custom_pretrain_path=model/pretrained/ckpt_mask_CS_100sup_top5.tar
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_300_CS_GTA_GTAcrops --net=lraspp_mobilenet --target_samples=100 --custom_pretrain_path=model/pretrained/ckpt_mask_lraspp_CS_GTA_300.tar
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_CS_600_scitas --net=lraspp_mobilenet --target_samples=100 --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_CS_100sup_allclasses --net=lraspp_mobilenet --target_samples=100 --custom_pretrain_path=model/pretrained/ckpt_mask_CS_100sup_allclasses.tar
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=no_pt_SCITAS_GTA_crops --net=lraspp_mobilenet --target_samples=100  

# Rotation pretrained backbone
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=dummy2_100_same_size_rot_pt --net=lraspp_mobilenet --target_samples=100 --custom_pretrain_path=ckpt_rot_10k.tar

# Rotation
# python pretrain/rotation.py --batch_size=4 --lr=0.001 --project=rotation --expt_name=longer_run_lr_x10 --save_dir=pretrain/expts_rot/tmp_last --save_interval=2000 --steps=10001

# ----- SSDA -----
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=CR_prob_distr_mask_pt_CS_NO_GTA --net=lraspp_mobilenet --target_samples=100 --cr=prob_distr --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar
python main_SSDA.py --project=GTA_CS_tiny_seeds --expt_name=CR_oh_mask_pt_CS --seed=3 --cr=one_hot --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar
