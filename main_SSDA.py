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
from loss.consistency import consistency_reg, cr_multiple_augs
from loader.loaders import get_loaders
from evaluation.metrics import averageMeter, runningScore
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
    
    # Load data
    source_loader, target_loader, target_loader_unl, val_loader = get_loaders(args)
    
    # Load model
    model = get_model(args)
    model.cuda()
    model.train()
    pdb.set_trace()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995).to(model.device)

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
    time_meter_cr = averageMeter()
    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()
    source_ce_loss_meter = averageMeter()
    target_ce_loss_meter = averageMeter()
    cr_loss_meter = averageMeter()
    constrast_s_loss_meter = averageMeter()
    constrast_t_loss_meter = averageMeter()
    pseudo_lbl_meter = averageMeter()

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

        start_ts = time.time()
        model.train()


        optimizer.zero_grad()
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

        # CE
        loss_s = loss_fn(out_s, labels_s)
        loss_t = loss_fn(out_t, labels_t)

        # CR
        loss_cr, percent_pl, time_cr = 0, 0, 0
        if args.cr is not None:
            start_ts_cr = time.time()
            if step % (len(target_loader_unl)-1) == 0:
                data_iter_t_unl = iter(target_loader_unl)
            
            images_t_unl = next(data_iter_t_unl)

            if args.n_augmentations == 1:
                images_weak = images_t_unl[0].cuda()
                images_strong = images_t_unl[1].cuda()

                if args.dsbn:
                    with ema.average_parameters():
                        outputs_w = model(images_weak, 1*torch.ones(images_s.shape[0], dtype=torch.long))                   # (N, C, H, W)
                    outputs_strong = model(images_strong, 2*torch.ones(images_s.shape[0], dtype=torch.long))
                else:
                    with ema.average_parameters():
                        outputs_w = model(images_weak)     # (N, C, H, W)
                    outputs_strong = model(images_strong)

                if type(outputs_w) == OrderedDict:
                    out_w = outputs_w['out']
                    out_strong = outputs_strong['out']
                else:
                    out_w = outputs_w
                    out_strong = outputs_strong
                loss_cr, percent_pl = consistency_reg(args.cr, out_w, out_strong, args.tau)
            else:
                assert args.n_augmentations >= 1 and not args.dsbn
                cr_multiple_augs(args, images_t_unl, model) # TODO EMA support

            time_cr = time.time() - start_ts_cr
            
        # CL
        loss_cl_s, loss_cl_t = 0, 0
        if args.pixel_contrast and step >= args.warmup_steps:
            proj_s = outputs_s['proj']
            proj_t = outputs_t['proj']

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

        loss = loss_s + loss_t + args.lmbda * loss_cr + args.gamma * (loss_cl_s + loss_cl_t)
        
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        ema.update()
        pdb.set_trace()

        time_meter.update(time.time() - start_ts)
        time_meter_cr.update(time_cr)
        train_loss_meter.update(loss)
        source_ce_loss_meter.update(loss_s)
        target_ce_loss_meter.update(loss_t)
        cr_loss_meter.update(args.lmbda * loss_cr)
        constrast_s_loss_meter.update(args.gamma * loss_cl_s)
        constrast_t_loss_meter.update(args.gamma * loss_cl_t)
        pseudo_lbl_meter.update(percent_pl)

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

        step += 1
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
            })

            log_str = get_log_str(args, log_info, title='Training Log')
            print(log_str)
            wandb.log(rm_format(log_info))

            source_ce_loss_meter.reset()
            target_ce_loss_meter.reset()
            cr_loss_meter.reset()
            constrast_s_loss_meter.reset()
            constrast_t_loss_meter.reset()
            train_loss_meter.reset()
            
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
                    # DM. save model as wandb artifact
                    model_artifact = wandb.Artifact('best_model_{}'.format(step), type='model')
                    model_artifact.add_file(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
                    wandb.log_artifact(model_artifact)
                best_mIoU = score['mIoU']
            
            
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
# python main_SSDA.py --net=deeplabv3_rn50 --pixel_contrast=True
