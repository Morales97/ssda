import os
import random
import shutil
from collections import OrderedDict
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import get_model
#from utils.eval import test
from utils.ioutils import FormattedLogItem
from utils.ioutils import gen_unique_name
from utils.ioutils import get_log_str
from utils.ioutils import parse_args
from utils.ioutils import rm_format
from loader.loaders import get_loaders
from evaluation.metrics import averageMeter, runningScore
import wandb
from loader.cityscapes_ds import cityscapesDataset
from torch_ema import ExponentialMovingAverage # https://github.com/fadel/pytorch_ema 


import pdb

# 0. Load validation loader
# 1. Load model
# 2. Evaluate


def evaluate(args, path_to_model):

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # --- Validation loader ---
    size = args.size
    assert size in ['tiny', 'small']

    if size == 'tiny':
        image_path_cs = 'data/cityscapes/leftImg8bit_tiny'
    elif size == 'small':
        image_path_cs = 'data/cityscapes/leftImg8bit_small'
    label_path_cs = 'data/cityscapes/gtFine'

    # NOTE downsampling or not the ground truth is found to make very little difference on the accuracy reported
    # However, it is x4 slower if downsample_gt = False
    val_dataset = cityscapesDataset(image_path=image_path_cs, label_path=label_path_cs, size=size, split='val', downsample_gt=True, hflip=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=1,
        shuffle=False,
    )    
    
    # --- Model ---
    model = get_model(args)
    model.cuda()
    ema_model = get_model(args)
    ema_model.cuda()

    if os.path.isfile(path_to_model):
        checkpoint = torch.load(path_to_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        step = checkpoint['step']
        print('Loading model trained until step {}'.format(step))
    else:
        raise Exception('No file found at {}'.format(path_to_model))
    
    # --- Evaluate ---
    running_metrics_val = runningScore(val_loader.dataset.n_classes)

    # evaluate on student
    model.eval()
    with torch.no_grad():
        for (images_val, labels_val) in val_loader:
            images_val = images_val.cuda()
            labels_val = labels_val.cuda()


            outputs = model(images_val)
            outputs = outputs['out']

            outputs = F.interpolate(outputs, size=(labels_val.shape[1], labels_val.shape[2]), mode="bilinear", align_corners=True)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            running_metrics_val.update(gt, pred)

    log_info = OrderedDict({
        'Train Step': step,
        #'Validation loss': val_loss_meter.avg
    })
    
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        log_info.update({k: FormattedLogItem(v, '{:.6f}')})

    #for k, v in class_iou.items():
    #    log_info.update({str(k): FormattedLogItem(v, '{:.6f}')})

    log_str = get_log_str(args, log_info, title='Validation Log')
    print('model on model.eval()')
    print(log_str)
    #wandb.log(rm_format(log_info))


    # evaluate on teacher
    running_metrics_val = runningScore(val_loader.dataset.n_classes)
    ema_model.eval()
    with torch.no_grad():
        for (images_val, labels_val) in val_loader:
            images_val = images_val.cuda()
            labels_val = labels_val.cuda()


            outputs = ema_model(images_val)
            outputs = outputs['out']

            outputs = F.interpolate(outputs, size=(labels_val.shape[1], labels_val.shape[2]), mode="bilinear", align_corners=True)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            running_metrics_val.update(gt, pred)

    log_info = OrderedDict({
        'Train Step': step,
        #'Validation loss': val_loss_meter.avg
    })
    
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        log_info.update({k: FormattedLogItem(v, '{:.6f}')})

    #for k, v in class_iou.items():
    #    log_info.update({str(k): FormattedLogItem(v, '{:.6f}')})

    log_str = get_log_str(args, log_info, title='Validation Log')
    print('model on model.eval()')
    print(log_str)
    #wandb.log(rm_format(log_info))


    

if __name__ == '__main__':
    args = parse_args()

    path_to_model='expts/tmp_last/checkpoint_full_r2_p2_3.pth.tar'  # round 2
    #path_to_model='expts/tmp_last/checkpoint_full_r3_p2_3.pth.tar' # round 3
    evaluate(args, path_to_model)
    
    print('** Round 2 **')

# python evaluate.py 