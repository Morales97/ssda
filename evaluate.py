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

    for k, v in class_iou.items():
        log_info.update({str(k): FormattedLogItem(v, '{:.6f}')})

    log_str = get_log_str(args, log_info, title='Validation Log')
    print('teacher EMA model')
    print(log_str)
    #wandb.log(rm_format(log_info))


def ensemble(args, path_1, path_2, path_3=None, viz_prediction=False):
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


    batch_size = 8
    if viz_prediction: batch_size = 1

    # NOTE downsampling or not the ground truth is found to make very little difference on the accuracy reported
    # However, it is x4 slower if downsample_gt = False
    val_dataset = cityscapesDataset(image_path=image_path_cs, label_path=label_path_cs, size=size, split='val', downsample_gt=True, hflip=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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
        #ema_model_1.load_state_dict(checkpoint['ema_state_dict'])
        ema_model_1.load_state_dict(checkpoint['model_state_dict'])     # NOTE better using student than teacher
        step = checkpoint['step']
        print('Loading model trained until step {}'.format(step))
    else:
        raise Exception('No file found at {}'.format(path_1))
    
    if os.path.isfile(path_2):
        checkpoint = torch.load(path_2)
        #ema_model_2.load_state_dict(checkpoint['ema_state_dict'])
        ema_model_2.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint['step']
        print('Loading model trained until step {}'.format(step))
    else:
        raise Exception('No file found at {}'.format(path_2))
    
    if path_3 is not None:
        ema_model_3 = get_model(args)
        ema_model_3.cuda()   

        if os.path.isfile(path_3):
                checkpoint = torch.load(path_3)
                ema_model_3.load_state_dict(checkpoint['ema_state_dict'])
                step = checkpoint['step']
                print('Loading model trained until step {}'.format(step))
        else:
            raise Exception('No file found at {}'.format(path_3))

    # -- MERGE models predictions--

    running_metrics_val = runningScore(val_loader.dataset.n_classes)
    running_metrics_val_1 = runningScore(val_loader.dataset.n_classes)
    running_metrics_val_2 = runningScore(val_loader.dataset.n_classes)
    ema_model_1.eval()
    ema_model_2.eval()
    with torch.no_grad():
        for i, (images_val, labels_val) in enumerate(val_loader):
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

            if path_3 is None:
                prob_ens = (prob_1 + prob_2)/2 # ensemble by combining probabilities
            else:
                outputs_3 = ema_model_3(images_val)
                outputs_3 = outputs_3['out']
                outputs_3 = F.interpolate(outputs_3, size=(labels_val.shape[1], labels_val.shape[2]), mode="bilinear", align_corners=True)
                prob_3 = F.softmax(outputs_3, dim=1)
                prob_ens = (prob_1 + prob_2 + prob_3)/3 # ensemble by combining probabilities

            pred = prob_ens.data.max(1)[1].cpu().numpy()
            #pred_1 = prob_1.data.max(1)[1].cpu().numpy()
            #pred_2 = prob_2.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            if viz_prediction:# and (i%20==0):
                val_dataset.save_pred_viz(pred, index=i, img_name='val_' + str(i), img=images_val, lbl=labels_val)
                if i == 9:
                    break

            running_metrics_val.update(gt, pred)
            #running_metrics_val_1.update(gt, pred_1)
            #running_metrics_val_2.update(gt, pred_2)

    # ** Ensemble model
    log_info = OrderedDict({
        'Train Step': step,
        #'Validation loss': val_loss_meter.avg
    })
    
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        log_info.update({k: FormattedLogItem(v, '{:.6f}')})

    for k, v in class_iou.items():
        log_info.update({str(k): FormattedLogItem(v, '{:.6f}')})

    log_str = get_log_str(args, log_info, title='Validation Log')
    print('Ensemble model')
    print(log_str)

    return score, class_iou
    '''
    # ** Round 1 model
    log_info = OrderedDict({
        'Train Step': step,
        #'Validation loss': val_loss_meter.avg
    })
    
    score, class_iou = running_metrics_val_1.get_scores()
    for k, v in score.items():
        log_info.update({k: FormattedLogItem(v, '{:.6f}')})

    log_str = get_log_str(args, log_info, title='Validation Log')
    print('Round 1 model')
    print(log_str)

    # ** Round 2 model
    log_info = OrderedDict({
        'Train Step': step,
        #'Validation loss': val_loss_meter.avg
    })
    
    score, class_iou = running_metrics_val_2.get_scores()
    for k, v in score.items():
        log_info.update({k: FormattedLogItem(v, '{:.6f}')})

    log_str = get_log_str(args, log_info, title='Validation Log')
    print('Round 2 model')
    print(log_str)
    '''

if __name__ == '__main__':
    args = parse_args()

    '''
    path_to_model='expts/tmp_last/checkpoint_full_rampupFIX_p2_3.pth.tar'  # round 1
    #path_to_model='expts/tmp_last/checkpoint_full_r2_p2_3.pth.tar'  # round 2
    #path_to_model='expts/tmp_last/checkpoint_full_r3_p2_3.pth.tar' # round 3
    evaluate(args, path_to_model)
    print('** Round 1 **')
    '''

    scores = []
    class_ious = []
    for seed in [1,2,3]:
        path_to_model_r2='expts/tmp_last/checkpoint_full_100_r2_' + str(seed) + '.pth.tar'  # round 2
        path_to_model_r3='expts/tmp_last/checkpoint_full_100_r3_' + str(seed) + '.pth.tar'  # round 3
        
        print('seed ', str(seed))
        score, class_iou = ensemble(args, path_to_model_r2, path_to_model_r3, viz_prediction=False)
        scores.append(score)
        class_ious.append(class_iou)

    n_seeds = len(class_ious)
    for key in class_ious[0].keys():
        avg = 0
        for i in range(n_seeds):
            avg += class_ious[i][key]
        avg /= n_seeds
        print(str(key) + '\t\t' + str(avg))


# python evaluate.py 