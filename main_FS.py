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
from loss.cross_entropy import cross_entropy2d
from loss.pixel_contrast import PixelContrastLoss
from loader.loaders import get_loaders
from loss.consistency import consistency_reg
from evaluation.metrics import averageMeter, runningScore
import wandb

import pdb


def main(args, wandb):
    torch.set_num_threads(args.max_num_threads)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load data
    args.target_samples = -1
    _, target_loader, _, val_loader = get_loaders(args)
    
    # Load model
    model = get_model(args)
    model.cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                            weight_decay=args.wd, nesterov=True)

    # To resume from a checkpoint
    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step'] + 1
            print('Resuming from train step {}'.format(start_step))
        else:
            raise Exception('No file found at {}'.format(args.resume))

    # Custom loss function. Ignores index 250
    loss_fn = cross_entropy2d   
    if args.pixel_contrast:
        pixel_contrast = PixelContrastLoss()

    # Set up metrics
    running_metrics_val = runningScore(target_loader.dataset.n_classes)
    best_mIoU = 0 
    step = start_step
    time_meter = averageMeter()
    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()

    # Training loop
    while step <= args.steps:

        if step % len(target_loader) == 0:
            data_iter_t = iter(target_loader)

        images_t, labels_t = next(data_iter_t)
        images_t = images_t.cuda()
        labels_t = labels_t.cuda()

        print(images_t.shape)
        #if images_t.shape[0] != 2:
        #    pdb.set_trace()
        continue

        start_ts = time.time()
        model.train()


        optimizer.zero_grad()
        outputs_t = model(images_t)
        if type(outputs_t) == OrderedDict:
            out_t = outputs_t['out']  
        else:
            out_t = outputs_t

        # CE
        loss = loss_fn(out_t, labels_t)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        step += 1

        time_meter.update(time.time() - start_ts)
        train_loss_meter.update(loss)

        # decrease lr
        if args.lr_decay == 'poly' and step % args.log_interval == 0:
            lr_min = args.lr/50
            lr = (args.lr - lr_min) * pow(1 - step/args.steps, 0.9) + lr_min
            print('*** Learning rate set to %.6f ***' % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif args.lr_decay == 'det':
            if step == np.floor(args.steps * 0.75):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

    
        if step % args.log_interval == 0:
            log_info = OrderedDict({
                'Train Step': step,
                'Time/Image [s]': round(time_meter.avg / args.batch_size_s, 3),
                'Train Loss': FormattedLogItem(train_loss_meter.avg, '{:.3f}'),
                'Norm in last update': FormattedLogItem(norm, '{:.4f}'),
            })

            log_str = get_log_str(args, log_info, title='Training Log')
            print(log_str)
            wandb.log(rm_format(log_info))

            train_loss_meter.reset()
            
        if step % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for (images_val, labels_val) in val_loader:
                    images_val = images_val.cuda()
                    labels_val = labels_val.cuda()

                    outputs = model(images_val)
                    if type(outputs) == OrderedDict:
                        outputs = outputs['out']
                    val_loss = loss_fn(input=outputs, target=labels_val)

                    pred = outputs.data.max(1)[1].cpu().numpy()
                    gt = labels_val.data.cpu().numpy()

                    running_metrics_val.update(gt, pred)
                    val_loss_meter.update(val_loss.item())

            log_info = OrderedDict({
                'Train Step': step,
                'Validation loss': val_loss_meter.avg
            })
            
            score, class_iou = running_metrics_val.get_scores()
            for k, v in score.items():
                log_info.update({k: FormattedLogItem(v, '{:.6f}')})

            for k, v in class_iou.items():
                log_info.update({str(k): FormattedLogItem(v, '{:.6f}')})

            log_str = get_log_str(args, log_info, title='Validation Log')
            print(log_str)
            wandb.log(rm_format(log_info))

            val_loss_meter.reset()
            running_metrics_val.reset()


        if step % args.save_interval == 0:
            if args.save_model:
                torch.save({
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'step' : step,
                }, os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            
            if score['mIoU'] > best_mIoU:
                if args.save_model:
                    shutil.copyfile(
                        os.path.join(args.save_dir, 'checkpoint.pth.tar'),
                        os.path.join(args.save_dir, 'model-best.pth.tar'))
                best_mIoU = score['mIoU']
            
                # DM. save model as wandb artifact
                model_artifact = wandb.Artifact('best_model_{}'.format(step), type='model')
                model_artifact.add_file(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
                wandb.log_artifact(model_artifact)
            
        if step >= args.steps:
            break

        
if __name__ == '__main__':
    args = parse_args()

    # W&B logging setup
    #wandb = WandbWrapper(debug=~args.use_wandb)
    if not args.expt_name:
        args.expt_name = gen_unique_name()
    #wandb.init(name=args.expt_name, dir=args.save_dir, config=args, reinit=True, project=args.project, entity=args.entity)
    wandb=None
    os.makedirs(args.save_dir, exist_ok=True)
    main(args, wandb)
    #wandb.finish()
    
# python main_SSDA.py --net=lraspp_mobilenet --target_samples=100 --batch_size=8 --cr=one_hot 
# python main_SSDA.py --net=lraspp_mobilenet_contrast --pixel_contrast=True
# python main_SSDA.py --net=lraspp_mobilenet_contrast --pixel_contrast=True --gamma=0.1 --pre_trained=True