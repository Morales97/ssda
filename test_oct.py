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

from model.resnet import resnet50_FCN, resnet_34_upsampling, resnet_50_upsampling, deeplabv3_rn50, deeplabv3_mobilenetv3_large, lraspp_mobilenetv3_large
from model.fcn import fcn8s
#from utils.eval import test
from utils.ioutils import FormattedLogItem
from utils.ioutils import gen_unique_name
from utils.ioutils import get_log_str
from utils.ioutils import parse_args
from utils.ioutils import rm_format
from loader.oct_loader import octLoader
from loader.gta_loader import gtaLoader
from loader.cityscapes_loader import cityscapesLoader
from loss.loss import cross_entropy2d
from evaluation.metrics import averageMeter, runningScore
import wandb
import tqdm

import pdb

def main(args, wandb):
    torch.set_num_threads(args.max_num_threads)

    # set random seed
    torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    random.seed(args.seed)

    #t_loader = octLoader(image_path='data/retouch-dataset/pre_processed/Spectralis_part1', label_path='data/retouch-dataset/pre_processed/Spectralis_part1', img_size=(512, 512))
    #t_loader = octLoader(image_path='data/retouch-dataset/pre_processed/Cirrus_part1', label_path='data/retouch-dataset/pre_processed/Cirrus_part1', img_size=(512, 512))
    v_loader = cityscapesLoader(image_path='data/cityscapes/leftImg8bit_tiny', label_path='data/cityscapes/gtFine', img_size=(256, 512), split='val')
    #v_loader.test()
    #s_loader = gtaLoader(image_path='data/gta5/images_tiny', label_path='data/gta5/labels', img_size=(360, 680))
    #s_loader = gtaLoader(image_path='data/images', label_path='data/labels', img_size=(360, 680))
    #s_loader.test()

    loader = DataLoader(
        v_loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # TODO change to True
    )

    for (images, labels) in loader:
        pdb.set_trace()

if __name__ == '__main__':
    args = parse_args()
    wandb = None
    main(args, wandb)
