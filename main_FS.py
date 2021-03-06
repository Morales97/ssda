import os
import random
import shutil
from collections import OrderedDict
import time
import copy
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import get_model
from utils.ioutils import gen_unique_name, get_log_str, parse_args, rm_format, FormattedLogItem
from loss.cross_entropy import cross_entropy2d
from loader.loaders import get_loaders, get_loaders_pseudolabels
from evaluation.metrics import averageMeter, runningScore
from utils.lab_color import lab_transform
from utils.class_balance import get_class_weights, get_class_weights_estimation
from utils.cutmix import _cutmix, _cutmix_output
import wandb

from torchvision.utils import save_image
import pdb


def main(args, wandb):
    torch.set_num_threads(args.max_num_threads)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print('Seed: ', args.seed)
    
    # Load data
    args.target_samples = -1
    _, target_loader, _, val_loader, _ = get_loaders(args)

    # Init model and EMA
    model = get_model(args)
    model.cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

    class_weigth_t = None
    if args.class_weight:
        class_weigth_t = get_class_weights(target_loader)

    # Custom loss function. Ignores index 250
    loss_fn = cross_entropy2d   

    # To resume from a checkpoint
    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
            print('*** Loading checkpoint from ', args.resume)
            print('*** Resuming from train step {}'.format(start_step))
            score = _log_validation(model, val_loader, loss_fn, start_step, wandb)

        else:
            raise Exception('No file found at {}'.format(args.resume))
    step = start_step

    # To split training in multiple consecutive jobs
    if args.steps_job == 0:
        job_step_limit = args.steps
    else:
        job_step_limit = start_step + args.steps_job    # set the maximum steps in this job

    # Set up metrics
    best_mIoU = 0 
    time_meter = averageMeter()
    time_meter_cr = averageMeter()
    time_meter_update = averageMeter()
    train_loss_meter = averageMeter()
    source_ce_loss_meter = averageMeter()
    target_ce_loss_meter = averageMeter()
    cr_loss_meter = averageMeter()
    constrast_s_loss_meter = averageMeter()
    constrast_t_loss_meter = averageMeter()
    pseudo_lbl_meter = averageMeter()
    alonso_contrast_meter = averageMeter()
    entropy_meter = averageMeter()

    data_iter_t = iter(target_loader)

    # Training loop
    while step <= args.steps:

        # This condition checks that the iterator has reached its end. len(loader) returns the number of batches
        if step % (len(target_loader)-1) == 0:
            data_iter_t = iter(target_loader)

        images_t, labels_t = next(data_iter_t)
        images_t = images_t.cuda()
        labels_t = labels_t.cuda()

        start_ts = time.time()
        model.train()

        # Forward pass
        outputs_t = model(images_t)
        out_t = outputs_t['out']  

        # *** Cross Entropy ***
        loss = loss_fn(out_t, labels_t, weight=class_weigth_t)

        # Update
        start_ts_update = time.time()
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        time_update = time.time() - start_ts_update

        # Meters
        time_meter.update(time.time() - start_ts)
        time_meter_update.update(time_update)
        train_loss_meter.update(loss)
        source_ce_loss_meter.update(0)
        target_ce_loss_meter.update(loss)
        cr_loss_meter.update(0)
        constrast_s_loss_meter.update(0)
        constrast_t_loss_meter.update(0)
        pseudo_lbl_meter.update(0)
        alonso_contrast_meter.update(0)
        entropy_meter.update(0)

        # Decrease lr
        if args.lr_decay == 'poly' and step % args.log_interval == 0:
            lr_min = args.lr/50
            lr = (args.lr - lr_min) * pow(1 - step/args.steps, 0.9) + lr_min
            print('*** Learning rate set to %.6f ***' % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif args.lr_decay == 'det':
            if step == np.floor(args.steps * 0.75):
                print('*** Learning rate set to %.6f ***' % (args.lr * 0.1))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

        step += 1
        # Log Training
        if step % args.log_interval == 0:
            log_info = OrderedDict({
                'Train Step': step,
                'Time/Image [s]': round(time_meter.avg / args.batch_size_s, 3),
                'Time CR/Image [s]': round(time_meter_cr.avg / args.batch_size_tu, 3),
                'Time update/Image [s]': round(time_meter_update.avg / args.batch_size_tu, 3),
                'CE Source Loss': FormattedLogItem(source_ce_loss_meter.avg, '{:.3f}'),
                'CE Target Loss': FormattedLogItem(target_ce_loss_meter.avg, '{:.3f}'),
                'CR Loss': FormattedLogItem(cr_loss_meter.avg, '{:.3f}'),
                'Constrast S Loss': FormattedLogItem(constrast_s_loss_meter.avg, '{:.3f}'),
                'Constrast T Loss': FormattedLogItem(constrast_t_loss_meter.avg, '{:.3f}'),
                'Train Loss': FormattedLogItem(train_loss_meter.avg, '{:.3f}'),
                'Pseudo lbl %': FormattedLogItem(pseudo_lbl_meter.avg, '{:.2f}'),
                'Norm in last update': FormattedLogItem(norm, '{:.4f}'),
                'Alonso CL Loss': FormattedLogItem(alonso_contrast_meter.avg, '{:.3f}'),
                'Entropy Loss': FormattedLogItem(entropy_meter.avg, '{:.3f}'),
            })

            log_str = get_log_str(args, log_info, title='Training Log')
            print(log_str)
            wandb.log(rm_format(log_info))

            time_meter.reset()
            time_meter_cr.reset()
            time_meter_update.reset()
            source_ce_loss_meter.reset()
            target_ce_loss_meter.reset()
            cr_loss_meter.reset()
            constrast_s_loss_meter.reset()
            constrast_t_loss_meter.reset()
            alonso_contrast_meter.reset()
            entropy_meter.reset()
            train_loss_meter.reset()
            
        # Log Validation
        if step % args.val_interval == 0:
            score = _log_validation(model, val_loader, loss_fn, step, wandb)
        
        # Save checkpoint
        if step % args.save_interval == 0:
            ckpt_name = 'checkpoint_' + args.expt_name + '_' + str(args.seed) + '.pth.tar'
            if args.save_model:
                torch.save({
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'step' : step,
                }, os.path.join(args.save_dir, ckpt_name))
                print('Checkpoint saved.')
                
                # DM. save model as wandb artifact
                model_artifact = wandb.Artifact('model_{}'.format(step), type='model')
                model_artifact.add_file(os.path.join(args.save_dir, ckpt_name))
                wandb.log_artifact(model_artifact)

            if score['mIoU'] > best_mIoU:
                if args.save_model:
                    shutil.copyfile(
                        os.path.join(args.save_dir, ckpt_name),
                        os.path.join(args.save_dir, 'model-best.pth.tar'))
                best_mIoU = score['mIoU']
            
            
        if step >= job_step_limit or step >= args.steps:

            # Save checkpoint
            ckpt_name = 'checkpoint_' + args.expt_name + '_' + str(args.seed) + '.pth.tar'
            if args.save_model:
                torch.save({
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'step' : step,
                }, os.path.join(args.save_dir, ckpt_name))
                print('Checkpoint saved.')
            break


def _forward_cr(args, model, ema_model, images_weak, images_strong):
    # Get psuedo-targets 'out_w'
    outputs_w = ema_model(images_weak)     # (N, C, H, W)    
    out_w = outputs_w['out']

    if args.cutmix_cr:                     # Apply CutMix to strongly augmented images (between them) and to their pseudo-targets
        images_strong, out_w = _cutmix_output(args, images_strong, out_w)

    outputs_strong = model(images_strong)
    out_strong = outputs_strong['out']
    return out_w, out_strong

def _log_validation(model, val_loader, loss_fn, step, wandb):
    running_metrics_val = runningScore(val_loader.dataset.n_classes)
    val_loss_meter = averageMeter()
    model.eval()
    with torch.no_grad():
        for (images_val, labels_val) in val_loader:
            images_val = images_val.cuda()
            labels_val = labels_val.cuda()

            outputs = model(images_val)
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

    #for k, v in class_iou.items():
    #    log_info.update({str(k): FormattedLogItem(v, '{:.6f}')})

    log_str = get_log_str(args, log_info, title='Validation Log')
    print(log_str)
    wandb.log(rm_format(log_info))

    return score

if __name__ == '__main__':
    args = parse_args()
    os.environ['WANDB_CACHE_DIR'] = '/scratch/izar/danmoral/.cache/wandb' # save artifacts in scratch workspace, deleted every 24h

    print('\nSSDA for semantic segmentation. Source: GTA, Target: Cityscapes\n')

    if not args.expt_name:
        args.expt_name = gen_unique_name()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.wandb:
        wandb.init(name=args.expt_name, dir=args.save_dir, config=args, reinit=True, project=args.project, entity=args.entity)
        main(args, wandb)     
        wandb.finish()
    else:
        main(args, None)


# python main_SSDA_EMA.py --wandb=False 

# next round of ST
#python main_SSDA.py --seed=1 --wandb=False --size=small --expt_name=KL_pc_round2 --cr=kl --pixel_contrast=True --pc_mixed=True --warmup_steps=0 --pseudolabel_folder=test_pl1_test --round_start=model/pretrained/checkpoint_KLE1_p2.pth.tar

