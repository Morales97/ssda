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


def evaluate(args):

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
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    ema.to(torch.device('cuda:' +  str(torch.cuda.current_device())))

    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'ema_state_dict' in checkpoint.keys():
            ema.load_state_dict(checkpoint['ema_state_dict'], strict=False)
        step = checkpoint['step'] + 1
        print('Loading model trained until step {}'.format(step))
    else:
        raise Exception('No file found at {}'.format(args.resume))
    
    # --- Evaluate ---
    running_metrics_val = runningScore(val_loader.dataset.n_classes)
    model.eval()
    with torch.no_grad():
        for (images_val, labels_val) in val_loader:
            images_val = images_val.cuda()
            labels_val = labels_val.cuda()

            if args.eval_ema:
                print('Evaluating on EMA teacher')
                with ema.average_parameters():
                    if args.dsbn:
                        outputs = model(images_val, 1*torch.ones(images_val.shape[0], dtype=torch.long))
                    else:
                        outputs = model(images_val)
            else:
                if args.dsbn:
                    outputs = model(images_val, 1*torch.ones(images_val.shape[0], dtype=torch.long))
                else:
                    outputs = model(images_val)

            if type(outputs) == OrderedDict:
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
    print(log_str)
    #wandb.log(rm_format(log_info))



if __name__ == '__main__':

    # W&B logging setup
    #wandb = WandbWrapper(debug=~args.use_wandb)
    #wandb.init(name=args.expt_name, dir=args.save_dir, config=args, reinit=True, project=args.project, entity=args.entity)
    #wandb=None
    #os.makedirs(args.save_dir, exist_ok=True)

    args = parse_args()

    evaluate(args)
    #wandb.finish()

# python evaluate.py --net=deeplabv3_rn50 --resume=model/pretrained/ckpt_15k_FS_small.tar --size=small
# python evaluate.py --net=deeplabv3_rn50 --resume=model/pretrained/ckpt_30k_FS_small.tar --size=small
# python evaluate.py --net=deeplabv2_rn101 --resume=model/pretrained/SS_kle_40k.tar --size=tiny --eval_ema=True