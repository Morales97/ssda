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
    print('student model')
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
    print('teacher EMA model')
    print(log_str)
    #wandb.log(rm_format(log_info))


def test_ensemble(args, path_1, path_2):
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
    ema_model_1 = get_model(args)
    ema_model_1.cuda()
    ema_model_2 = get_model(args)
    ema_model_2.cuda()    

    if os.path.isfile(path_1):
        checkpoint = torch.load(path_1)
        ema_model_1.load_state_dict(checkpoint['ema_state_dict'])
        step = checkpoint['step']
        print('Loading model trained until step {}'.format(step))
    else:
        raise Exception('No file found at {}'.format(path_1))
    
    if os.path.isfile(path_2):
        checkpoint = torch.load(path_2)
        ema_model_2.load_state_dict(checkpoint['ema_state_dict'])
        step = checkpoint['step']
        print('Loading model trained until step {}'.format(step))
    else:
        raise Exception('No file found at {}'.format(path_2))
    
    # -- MERGE models predictions--

    running_metrics_val = runningScore(val_loader.dataset.n_classes)
    ema_model_1.eval()
    ema_model_2.eval()
    with torch.no_grad():
        for (images_val, labels_val) in val_loader:
            images_val = images_val.cuda()
            labels_val = labels_val.cuda()

            outputs_1 = ema_model_1(images_val)
            outputs_1 = outputs_1['out']
            outputs_1 = F.interpolate(outputs_1, size=(labels_val.shape[1], labels_val.shape[2]), mode="bilinear", align_corners=True)
            prob_1 = F.softmax(outputs_1, dim=1)

            outputs_2 = ema_model_2(images_val)
            outputs_2 = outputs_2['out']
            outputs_2 = F.interpolate(outputs_2, size=(labels_val.shape[1], labels_val.shape[2]), mode="bilinear", align_corners=True)
            prob_2 = F.softmax(outputs_2, dim=1)

            prob_ens = (prob_1 + prob_2)/2 # ensemble by combining probabilities


            pred = prob_ens.data.max(1)[1].cpu().numpy()
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
    print('Ensemble model')
    print(log_str)


if __name__ == '__main__':
    args = parse_args()
    '''
    #path_to_model='expts/tmp_last/checkpoint_full_r2_p2_3.pth.tar'  # round 2
    path_to_model='expts/tmp_last/checkpoint_full_r3_p2_3.pth.tar' # round 3
    evaluate(args, path_to_model)
    print('** Round 3 **')
    '''
    path_to_model_r2='expts/tmp_last/checkpoint_full_r2_p2_3.pth.tar'  # round 2
    path_to_model_r3='expts/tmp_last/checkpoint_full_r3_p2_3.pth.tar' # round 3
    
    test_ensemble(args, path_to_model_r2, path_to_model_r3)


# python evaluate.py 