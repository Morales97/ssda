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
#SBATCH --time=14:00:00

# ----- Fully supervised CS -----
#python main.py --project=GTA_to_CS_tiny --expt_name=deeplab_rn50_FS_CS_pt_mask --net=deeplabv3_mask_pt --steps=5000
#python main.py --project=GTA_to_CS_tiny --expt_name=only_100_CS --net=lraspp_mobilenet --steps=5000 --target_samples=100 

# ----- Dummy 100 -----
# -- DeepLab --
# python main_dummyDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_PASCAL_lr3 --net=deeplabv3_mask_pt --target_samples=100 --custom_pretrain_path=model/pretrained/VOCSegmentation_supervised_saliency_model.pth.tar 
# -- LR-ASPP --
#python main_SSDA.py --project=GTA_CS_tiny_seeds --expt_name=mask_pt_CS_GTA_sup_s1 --seed=1 --custom_pretrain_path=model/pretrained/ckpt_mask_lraspp_CS_GTA_sup_s1.tar
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=mask_pt_CS_only100sup --net=lraspp_mobilenet --target_samples=100  --custom_pretrain_path=model/pretrained/ckpt_masks_lraspp_CS_only100sup.tar

# Rotation pretrained backbone
#python main_dummyDA.py --project=GTA_to_CS_tiny --expt_name=dummy2_100_same_size_rot_pt --net=lraspp_mobilenet --target_samples=100 --custom_pretrain_path=ckpt_rot_10k.tar

# Rotation
# python pretrain/rotation.py --batch_size=4 --lr=0.001 --project=rotation --expt_name=longer_run_lr_x10 --save_dir=pretrain/expts_rot/tmp_last --save_interval=2000 --steps=10001


# ----- SSDA -----
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=CR_no_pt_one_hot_lambda5 --net=lraspp_mobilenet --target_samples=100 --cr=one_hot --lmbda=5
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=CR_prob_distr_mask_pt_300_CS_GTA --net=lraspp_mobilenet --target_samples=100 --cr=prob_distr --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_GTA.pth.tar
#python main_SSDA.py --project=GTA_to_CS_tiny--expt_name=CL_no_warmup --net=lraspp_mobilenet_contrast --pixel_contrast=True --warmup_steps=0
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=CL_warmup --net=lraspp_mobilenet_contrast --pixel_contrast=True 
#python main_SSDA.py --project=test_seeds --expt_name=CL_CR_mask_pt_simba --seed=1 --net=lraspp_mobilenet_contrast --pixel_contrast=True 

# -- Only on CS --
#python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=CR_prob_distr_mask_pt_CS_NO_GTA --net=lraspp_mobilenet --target_samples=100 --batch_size_tl=16 --cr=prob_distr --custom_pretrain_path=model/pretrained/checkpoint_mask_lraspp_CS_600.pth.tar

# -- DeepLabV3 ---
python main_SSDA.py --project=GTA_to_CS_tiny --expt_name=CR_IN_bb --net=deeplabv3_rn50 --cr=prob_distr 
#python main_FS.py --lr=0.001 --steps=40000 --size=small --project=GTA_to_CS_small --expt_name=IN_bb_dl_rn50_FS --net=deeplabv3_rn50 --batch_size_tl=4
