import os
import random
import shutil
from collections import OrderedDict
import time
import copy
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
from loss.pixel_contrast_unsup import AlonsoContrastiveLearner
from loss.consistency import consistency_reg, cr_multiple_augs
from loader.loaders import get_loaders
from evaluation.metrics import averageMeter, runningScore
from utils.lab_color import lab_transform
import wandb
from torch_ema import ExponentialMovingAverage # https://github.com/fadel/pytorch_ema 
from utils.class_balance import get_class_weights

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
    _, target_loader, target_loader_unl, val_loader, _, _ = get_loaders(args)
    
    class_weigth_t = None
    if args.class_weight:
        class_weigth_t = get_class_weights(target_loader)

    # Load model
    model = get_model(args)
    model.cuda()
    model.train()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    ema.to(torch.device('cuda:' +  str(torch.cuda.current_device())))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                            weight_decay=args.wd, nesterov=True)

    # To resume from a checkpoint
    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'ema_state_dict' in checkpoint.keys():
                ema.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
            print('Resuming from train step {}'.format(start_step))
        else:
            raise Exception('No file found at {}'.format(args.resume))

    # Custom loss function. Ignores index 250
    loss_fn = cross_entropy2d   
    if args.pixel_contrast:
        pixel_contrast = PixelContrastLoss()
    if args.alonso_contrast is not None:
        alonso_pc_learner = AlonsoContrastiveLearner(args.alonso_contrast, args.target_samples)

    # Set up metrics
    running_metrics_val = runningScore(target_loader.dataset.n_classes)
    best_mIoU = 0 
    step = start_step
    time_meter = averageMeter()
    time_meter_cr = averageMeter()
    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()
    source_ce_loss_meter = averageMeter()
    target_ce_loss_meter = averageMeter()
    cr_loss_meter = averageMeter()
    constrast_s_loss_meter = averageMeter()
    constrast_t_loss_meter = averageMeter()
    pseudo_lbl_meter = averageMeter()
    alonso_contrast_meter = averageMeter()

    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)

    # Training loop
    while step <= args.steps:

        # This condition checks that the iterator has reached its end. len(loader) returns the number of batches
        if step % (len(target_loader)-1) == 0:
            data_iter_t = iter(target_loader)

        images_t, labels_t = next(data_iter_t)
        images_t = images_t.cuda()
        labels_t = labels_t.cuda()

        if args.cr is not None or args.alonso_contrast:
            if step % (len(target_loader_unl)-1) == 0:
                data_iter_t_unl = iter(target_loader_unl)
            
            images_t_unl = next(data_iter_t_unl)


        start_ts = time.time()
        model.train()

        # Forward pass
        optimizer.zero_grad()
        out_t, outputs_t = _forward(args, model, images_t)

        # *** Cross Entropy ***
        loss_s = 0
        loss_t = loss_fn(out_t, labels_t, weight=class_weigth_t)

        # *** Consistency Regularization ***
        loss_cr, percent_pl, time_cr = 0, 0, 0
        if args.cr is not None:
            start_ts_cr = time.time()

            if args.n_augmentations == 1:
                images_weak = images_t_unl[0].cuda()
                images_strong = images_t_unl[1].cuda()

                # Forward pass for CR
                out_w, out_strong = _forward_cr(args, model, ema, images_weak, images_strong, step)
                loss_cr, percent_pl = consistency_reg(args.cr, out_w, out_strong, args.tau)
            else:
                assert args.n_augmentations >= 1 and not args.dsbn
                cr_multiple_augs(args, images_t_unl, model) # TODO EMA support

            time_cr = time.time() - start_ts_cr
            
        # *** Pixel Contrastive Learning (supervised) ***
        loss_cl_s, loss_cl_t = 0, 0
        if args.pixel_contrast and step >= args.warmup_steps:
            proj_t = outputs_t['proj']
            _, pred_t = torch.max(out_t, 1)


            loss_cl_s = 0 
            loss_cl_t = pixel_contrast(proj_t, labels_t, pred_t)


        # *** Pixel Contrastive Learning (sup and unsupervised, Alonso et al) ***
        ramp_up_steps = 500
        loss_cl_alonso = 0

        if args.alonso_contrast is not None:
            '''
            options:
                - "base": without any selector
                - "full": use selectors heads in memory bank to save the highest quality and to weight the loss
            '''

            # Build feature memory bank, start 'ramp_up_steps' before
            if step >= args.warmup_steps - ramp_up_steps:
                with ema.average_parameters():
                    if args.dsbn:
                        outputs_t_ema = model(images_t, 1*torch.ones(images_t.shape[0], dtype=torch.long))  
                    else:
                        outputs_t_ema = model(images_t)   

                alonso_pc_learner.add_features_to_memory(outputs_t_ema, labels_t, model)

            # Contrastive Learning
            if step >= args.warmup_steps:
                # ** Labeled CL **
                # NOTE beware that it can compete with our PC!
                loss_labeled = alonso_pc_learner.labeled_pc(None, outputs_t, None, labels_t, model)

                # ** Unlabeled CL **
                images_tu = images_t_unl[0].cuda() # TODO change loader? rn unlabeled loader returns [weak, strong], for CR
                if args.dsbn:
                    outputs_tu = model(images_tu, 1*torch.ones(images_tu.shape[0], dtype=torch.long)) 
                else:
                    outputs_tu = model(images_tu)      # TODO merge this with forward in CR (this is the same forward pass)
                loss_unlabeled = alonso_pc_learner.unlabeled_pc(outputs_tu, model)

                loss_cl_alonso = loss_labeled + loss_unlabeled


        # Total Loss
        loss = loss_s + loss_t + args.lmbda * loss_cr + args.gamma * (loss_cl_s + loss_cl_t) + loss_cl_alonso
        
        # Update
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        ema.update()

        # Meters
        time_meter.update(time.time() - start_ts)
        time_meter_cr.update(time_cr)
        train_loss_meter.update(loss)
        source_ce_loss_meter.update(loss_s)
        target_ce_loss_meter.update(loss_t)
        cr_loss_meter.update(args.lmbda * loss_cr)
        constrast_s_loss_meter.update(args.gamma * loss_cl_s)
        constrast_t_loss_meter.update(args.gamma * loss_cl_t)
        pseudo_lbl_meter.update(percent_pl)
        alonso_contrast_meter.update(loss_cl_alonso)

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
                'CE Source Loss': FormattedLogItem(source_ce_loss_meter.avg, '{:.3f}'),
                'CE Target Loss': FormattedLogItem(target_ce_loss_meter.avg, '{:.3f}'),
                'CR Loss': FormattedLogItem(cr_loss_meter.avg, '{:.3f}'),
                'Constrast S Loss': FormattedLogItem(constrast_s_loss_meter.avg, '{:.3f}'),
                'Constrast T Loss': FormattedLogItem(constrast_t_loss_meter.avg, '{:.3f}'),
                'Train Loss': FormattedLogItem(train_loss_meter.avg, '{:.3f}'),
                'Pseudo lbl %': FormattedLogItem(pseudo_lbl_meter.avg, '{:.2f}'),
                'Norm in last update': FormattedLogItem(norm, '{:.4f}'),
                'Alonso CL Loss': FormattedLogItem(alonso_contrast_meter.avg, '{:.3f}'),
            })

            log_str = get_log_str(args, log_info, title='Training Log')
            print(log_str)
            wandb.log(rm_format(log_info))

            source_ce_loss_meter.reset()
            target_ce_loss_meter.reset()
            cr_loss_meter.reset()
            constrast_s_loss_meter.reset()
            constrast_t_loss_meter.reset()
            alonso_contrast_meter.reset()
            train_loss_meter.reset()
            
        # Log Validation
        if step % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for (images_val, labels_val) in val_loader:
                    images_val = images_val.cuda()
                    labels_val = labels_val.cuda()

                    if args.dsbn:
                        outputs = model(images_val, 1*torch.ones(images_val.shape[0], dtype=torch.long))
                    else:
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

            #for k, v in class_iou.items():
            #    log_info.update({str(k): FormattedLogItem(v, '{:.6f}')})

            log_str = get_log_str(args, log_info, title='Validation Log')
            print(log_str)
            wandb.log(rm_format(log_info))

            val_loss_meter.reset()
            running_metrics_val.reset()
        
        # Save checkpoint
        if step % args.save_interval == 0:
            ckpt_name = 'checkpoint_' + args.expt_name + '.pth.tar'
            if args.save_model:
                torch.save({
                    'model_state_dict' : model.state_dict(),
                    'ema_state_dict' : ema.state_dict(),
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
            
            
        if step >= args.steps:
            break


def _forward(args, model, images_t):
    outputs_t = model(images_t)

    if type(outputs_t) == OrderedDict:
        out_t = outputs_t['out']  
    else:
        out_t = outputs_t

    return out_t, outputs_t


def _forward_cr(args, model, ema, images_weak, images_strong, step):

    if step >= args.warmup_steps:
        with ema.average_parameters():
            outputs_w = model(images_weak)     # (N, C, H, W)
    else:
        outputs_w = model(images_weak)
    outputs_strong = model(images_strong)

    if type(outputs_w) == OrderedDict:
        out_w = outputs_w['out']
        out_strong = outputs_strong['out']
    else:
        out_w = outputs_w
        out_strong = outputs_strong

    return out_w, out_strong


if __name__ == '__main__':
    args = parse_args()
    os.environ['WANDB_CACHE_DIR'] = '/scratch/izar/danmoral/.cache/wandb' # save artifacts in scratch workspace, deleted every 24h

    print('\nSemi-Supervised semantic segmentation on Cityscapes\n')

    if not args.expt_name:
        args.expt_name = gen_unique_name()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.wandb:
        wandb.init(name=args.expt_name, dir=args.save_dir, config=args, reinit=True, project=args.project, entity=args.entity)
        main(args, wandb)     
        wandb.finish()
    else:
        main(args, None)
    

# python main_SemiSup.py --net=deeplabv3_rn50 --wandb=False --class_weight=True --alonso_contrast=True --warmup_step=0
