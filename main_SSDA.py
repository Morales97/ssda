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
from loss.pixel_contrast_unsup import FeatureMemory, contrastive_class_to_class
from loss.consistency import consistency_reg, cr_multiple_augs
from loader.loaders import get_loaders
from evaluation.metrics import averageMeter, runningScore
from utils.lab_color import lab_transform
import wandb
from torch_ema import ExponentialMovingAverage # https://github.com/fadel/pytorch_ema 

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
    source_loader, target_loader, target_loader_unl, val_loader = get_loaders(args)
    
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
    if args.alonso_contrast:
        feature_memory = FeatureMemory(num_samples=args.target_samples)

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

    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)

    # Training loop
    while step <= args.steps:

        # This condition checks that the iterator has reached its end. len(loader) returns the number of batches
        if step % (len(source_loader)-1) == 0:
            data_iter_s = iter(source_loader)
        if step % (len(target_loader)-1) == 0:
            data_iter_t = iter(target_loader)

        images_s, labels_s = next(data_iter_s)
        images_t, labels_t = next(data_iter_t)

        images_s = images_s.cuda()
        labels_s = labels_s.cuda()
        images_t = images_t.cuda()
        labels_t = labels_t.cuda()

        if args.lab_color:
            images_s = lab_transform(images_s, images_t)

        if args.cr is not None or args.alonso_contrast:
            if step % (len(target_loader_unl)-1) == 0:
                data_iter_t_unl = iter(target_loader_unl)
            
            images_t_unl = next(data_iter_t_unl)


        start_ts = time.time()
        model.train()

        # Forward pass
        optimizer.zero_grad()
        out_s, out_t, outputs_s, outputs_t = _forward(args, model, images_s, images_t)

        # *** Cross Entropy ***
        loss_s = loss_fn(out_s, labels_s)
        loss_t = loss_fn(out_t, labels_t)


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
            proj_s = outputs_s['proj_pc']
            proj_t = outputs_t['proj_pc']

            _, pred_s = torch.max(out_s, 1) 
            _, pred_t = torch.max(out_t, 1)

            if not args.pc_mixed:
                loss_cl_s = 0 #pixel_contrast(proj_s, labels_s, pred_s)
                loss_cl_t = pixel_contrast(proj_t, labels_t, pred_t)
            else:
                loss_cl_s = 0
                proj = torch.cat([proj_s, proj_t], dim=0)
                labels = torch.cat([labels_s, labels_t], dim=0)
                pred = torch.cat([pred_s, pred_t], dim=0)
                loss_cl_t = pixel_contrast(proj, labels, pred)


        # *** Pixel Contrastive Learning (sup and unsupervised, Alonso et al) ***
        ramp_up_steps = 500
        loss_cl_alonso = 0

        if args.alonso_contrast:

            # Build feature memory bank, start 'ramp_up_steps' before
            if step >= args.warmup_steps - ramp_up_steps:
                with ema.average_parameters() and torch.no_grad():  # NOTE if instead of using EMA we reuse out_s from CE (and detach() it), we might make it quite faster
                    outputs_t = model(images_t)   

                #prob_s, pred_s = torch.max(torch.softmax(outputs_s['out'], dim=1), dim=1)  
                prob_t, pred_t = torch.max(torch.softmax(outputs_t['out'], dim=1), dim=1)  

                # save the projected features if the prediction is correct and more confident than 0.95
                # the projected features are not upsampled, it is a lower resolution feature map. Downsample labels and preds (x8)
                #proj_s = outputs_s['proj']
                proj_t = outputs_t['proj']
                #labels_s_down = F.interpolate(labels_s.unsqueeze(0).float(), size=(proj_s.shape[2], proj_s.shape[3]), mode='nearest').squeeze()
                labels_t_down = F.interpolate(labels_t.unsqueeze(0).float(), size=(proj_t.shape[2], proj_t.shape[3]), mode='nearest').squeeze()
                pred_t_down = F.interpolate(pred_t.unsqueeze(0).float(), size=(proj_t.shape[2], proj_t.shape[3]), mode='nearest').squeeze()
                prob_t_down = F.interpolate(prob_t.unsqueeze(0), size=(proj_t.shape[2], proj_t.shape[3]), mode='nearest').squeeze()
                
                mask = ((pred_t_down == labels_t_down).float() * (prob_t_down > 0.95).float()).bool() # (B, 32, 64)
                labels_t_down_selected = labels_t_down[mask]

                proj_t = proj_t.permute(0,2,3,1)    # (B, 32, 64, C)
                proj_t_selected = proj_t[mask, :]
                
                if proj_t_selected.shape[0] > 0:
                    feature_memory.add_features(None, proj_t_selected, labels_t_down_selected, args.batch_size_tl)

                store_S_pixels = False  # Results are better when only storing features from T, not S+T. This is also what Alonso et al does
                if store_S_pixels:
                    with ema.average_parameters() and torch.no_grad():  
                        outputs_s = model(images_s) 

                    prob_s, pred_s = torch.max(torch.softmax(outputs_s['out'], dim=1), dim=1)  
                    proj_s = outputs_s['proj']
                    labels_s_down = F.interpolate(labels_s.unsqueeze(0).float(), size=(proj_s.shape[2], proj_s.shape[3]), mode='nearest').squeeze()
                    pred_s_down = F.interpolate(pred_s.unsqueeze(0).float(), size=(proj_s.shape[2], proj_s.shape[3]), mode='nearest').squeeze() 
                    prob_s_down = F.interpolate(prob_s.unsqueeze(0), size=(proj_s.shape[2], proj_s.shape[3]), mode='nearest').squeeze()

                    mask = ((pred_s_down == labels_s_down).float() * (prob_s_down > 0.95).float()).bool() # (B, 32, 64)
                    labels_s_down_selected = labels_s_down[mask]

                    proj_s = proj_s.permute(0,2,3,1)    # (B, 32, 64, C)
                    proj_s_selected = proj_s[mask, :]
                    
                    if proj_s_selected.shape[0] > 0:
                        feature_memory.add_features(None, proj_s_selected, labels_s_down_selected, args.batch_size_s)


            # Contrastive Learning
            if step >= args.warmup_steps:
                # Labeled CL 
                # NOTE not implemented (Alonso et al does). Can try to implement this - but beware that it can compete with our PC!

                # Unlabeled CL
                images_tu = images_t_unl[0].cuda() # TODO change loader? rn unlabeled loader returns [weak, strong], for CR
                outputs_tu = model(images_tu)      # TODO merge this with forward in CR (this is the same forward pass)
                pred_tu = outputs_tu['pred']

                # compute pseudolabel
                prob, pseudo_lbl = torch.max(F.softmax(outputs_tu['out'], dim=1).detach(), dim=1)
                pseudo_lbl_down = F.interpolate(pseudo_lbl.unsqueeze(0).float(), size=(pred_tu.shape[2], pred_tu.shape[3]), mode='nearest').squeeze()
                prob_down = F.interpolate(prob.unsqueeze(0), size=(pred_tu.shape[2], pred_tu.shape[3]), mode='nearest').squeeze()

                # take out the features from black pixels from zooms out and augmetnations 
                ignore_label = 250
                threshold = 0
                mask = prob_down > threshold
                mask = mask * (pseudo_lbl_down != ignore_label)    # this is legacy from Alonso et al, but might be useful if we introduce zooms and crops

                pred_tu = pred_tu.permute(0, 2, 3, 1)
                pred_tu = pred_tu[mask, ...]
                pseudo_lbl_down = pseudo_lbl_down[mask]

                loss_cl_alonso = contrastive_class_to_class(None, pred_tu, pseudo_lbl_down, feature_memory.memory)


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
            if args.save_model:
                torch.save({
                    'model_state_dict' : model.state_dict(),
                    'ema_state_dict' : ema.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'step' : step,
                }, os.path.join(args.save_dir, 'checkpoint.pth.tar'))
                print('Checkpoint saved.')
                
                # DM. save model as wandb artifact
                model_artifact = wandb.Artifact('model_{}'.format(step), type='model')
                model_artifact.add_file(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
                wandb.log_artifact(model_artifact)

            if score['mIoU'] > best_mIoU:
                if args.save_model:
                    shutil.copyfile(
                        os.path.join(args.save_dir, 'checkpoint.pth.tar'),
                        os.path.join(args.save_dir, 'model-best.pth.tar'))
                best_mIoU = score['mIoU']
            
            
        if step >= args.steps:
            break


def _forward(args, model, images_s, images_t):
    if args.dsbn:
        outputs_s = model(images_s, 0*torch.ones(images_s.shape[0], dtype=torch.long))
        outputs_t = model(images_t, 1*torch.ones(images_t.shape[0], dtype=torch.long))
    else:
        outputs_s = model(images_s)
        outputs_t = model(images_t)

    if type(outputs_t) == OrderedDict:
        out_s = outputs_s['out'] 
        out_t = outputs_t['out']  
    else:
        out_s = outputs_s
        out_t = outputs_t

    return out_s, out_t, outputs_s, outputs_t


def _forward_cr(args, model, ema, images_weak, images_strong, step):
    if args.dsbn:
        if step >= args.warmup_steps:
            with ema.average_parameters() and torch.no_grad():
                outputs_w = model(images_weak, 1*torch.ones(images_weak.shape[0], dtype=torch.long))                   # (N, C, H, W)
        else:
            outputs_w = model(images_weak, 1*torch.ones(images_weak.shape[0], dtype=torch.long))
        outputs_strong = model(images_strong, 1*torch.ones(images_strong.shape[0], dtype=torch.long))
    else:
        if step >= args.warmup_steps:
            with ema.average_parameters() and torch.no_grad():
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
    

# python main_SSDA.py --net=deeplabv3_rn50_densecl --wandb=False
# python main_SSDA.py --net=deeplabv3_rn50 --wandb=False --alonso_contrast=True
