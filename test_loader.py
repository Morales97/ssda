import os
import random
import shutil
from collections import OrderedDict
import time

#import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#from utils.eval import test
from utils.ioutils import FormattedLogItem
from utils.ioutils import gen_unique_name
from utils.ioutils import get_log_str
from utils.ioutils import parse_args
from utils.ioutils import rm_format
from loader.oct_loader import octLoader
from loader.gta_ds import gtaDataset
from loader.cityscapes_ds import cityscapesDataset

from loss.cross_entropy import cross_entropy2d
from evaluation.metrics import averageMeter, runningScore
import wandb
import tqdm
from model.model import get_model
from torch_ema import ExponentialMovingAverage
from loader.loaders import get_loaders, generate_pseudolabels
import pdb

def main(args, wandb):
    torch.set_num_threads(args.max_num_threads)

    # set random seed
    torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    random.seed(args.seed)

    #t_loader = octLoader(image_path='data/retouch-dataset/pre_processed/Spectralis_part1', label_path='data/retouch-dataset/pre_processed/Spectralis_part1', img_size=(512, 512))
    #t_loader = octLoader(image_path='data/retouch-dataset/pre_processed/Cirrus_part1', label_path='data/retouch-dataset/pre_processed/Cirrus_part1', img_size=(512, 512))
    '''t_ds = cityscapesDataset(image_path='data/cityscapes/leftImg8bit_tiny', label_path='data/cityscapes/gtFine', size="tiny", split='train', n_samples=100,
                            unlabeled=True, # to use weak-strong augs
                            strong_aug_level=5)'''
    #v_loader.test()
    #t_unl_ds = cityscapesDataset(image_path='data/cityscapes/leftImg8bit_tiny', label_path='data/cityscapes/gtFine', size="tiny", unlabeled=True, n_samples=args.target_samples)
    s_loader = gtaDataset(image_path='data/gta5/images_tiny', label_path='data/gta5/labels', size="tiny", split="val")
    s_loader.test()
    #t_unl_loader.test()

    loader = DataLoader(
        t_ds,   
        batch_size=args.batch_size_tl,
        num_workers=args.num_workers,
        shuffle=False, 
    )

    # for unlabeled loader
    for images in loader:
        img_weak = images[0]
        img_strong = images[1]
        save_image(img_weak[0], 'img_weak.jpg')
        save_image(img_strong[0], 'img_strong.jpg')

        pdb.set_trace()

    # for labeled loader
    for (images, labels) in loader:
        pdb.set_trace()

if __name__ == '__main__':
    args = parse_args()
    wandb = None
    print('starting...')
    #main(args, wandb)
    
    #path = 'model/pretrained/checkpoint_KLE1_p2.pth.tar'
    path = 'model/pretrained/model_40k_KL_pc.tar'
    args.net = 'deeplabv2_rn101'
    args.size = 'small'
    args.seed = 3  
    args.expt_name = 'KL_pc_40k_'

    model = get_model(args)
    model.cuda()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    ema.to(torch.device('cuda:' +  str(torch.cuda.current_device())))

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'ema_state_dict' in checkpoint.keys():
        ema.load_state_dict(checkpoint['ema_state_dict'])

    source_loader, target_loader, target_loader_unl, val_loader, target_lbl_dataset, target_unlbl_dataset = get_loaders(args)

    pseudolabel_folder = args.expt_name + str(args.seed)
    target_lbl_dataset.pseudolabel_folder = pseudolabel_folder
    target_unlbl_dataset.pseudolabel_folder = pseudolabel_folder
    os.makedirs('data/cityscapes/pseudo_labels/' + pseudolabel_folder, exist_ok=True)
    target_lbl_dataset.save_gt_labels()
    #target_unlbl_dataset.generate_pseudolabels(model, ema)


