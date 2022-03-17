import os
import random
import shutil
from collections import OrderedDict
import time

import numpy as np
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
from loader.cityscapes_loader import cityscapesLoader
from loader.gta_loader import gtaLoader
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

    # TODO: rn loaders don't use augmentations. Probably should be using some
    t_loader = cityscapesLoader(image_path='data/cityscapes/leftImg8bit_tiny', label_path='data/cityscapes/gtFine', img_size=(256, 512), split='train', few_samples=args.target_samples)
    v_loader = cityscapesLoader(image_path='data/cityscapes/leftImg8bit_tiny', label_path='data/cityscapes/gtFine', img_size=(256, 512), split='val')
    #t_loader = cityscapesLoader(image_path='data/cityscapes/leftImg8bit_small', label_path='data/cityscapes/gtFine', img_size=(256, 512), split='train', few_samples=args.target_samples)
    #v_loader = cityscapesLoader(image_path='data/cityscapes/leftImg8bit_small', label_path='data/cityscapes/gtFine', img_size=(256, 512), split='val')
    s_loader = gtaLoader(image_path='data/gta5/images_tiny', label_path='data/gta5/labels', size="tiny", split="all_gta")
    #t_loader = gtaLoader(image_path='data/gta5/images_tiny', label_path='data/gta5/labels', size="tiny", split="train")
    #v_loader = gtaLoader(image_path='data/gta5/images_tiny', label_path='data/gta5/labels', size="tiny", split="val")

    train_source_loader = DataLoader(
        s_loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    train_target_loader = DataLoader(
        t_loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_loader = DataLoader(
        v_loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )    

    # Set up metrics
    running_metrics_val = runningScore(t_loader.n_classes)

    # Init model
    if args.net == '' or args.net == 'resnet50_fcn':
        model = resnet50_FCN(args.pre_trained)
    if args.net == 'fcn8':
        model = fcn8s()
    if args.net == 'rn34_up':
        model = resnet_34_upsampling(args.pre_trained)
    if args.net == 'rn50_up':
        model = resnet_50_upsampling(args.pre_trained)
    if args.net == 'deeplabv3':
        model = deeplabv3_rn50(args.pre_trained, args.pre_trained_backbone)
    if args.net == 'dl_mobilenet':
        model = deeplabv3_mobilenetv3_large(args.pre_trained, args.pre_trained_backbone)
    if args.net == 'lraspp_mobilenet':
        model = lraspp_mobilenetv3_large(args.pre_trained, args.pre_trained_backbone)
    model.cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
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

    best_mIoU = 0 
    step = start_step
    time_meter = averageMeter()
    val_loss_meter = averageMeter()

    while step <= args.steps:
        for (images, labels) in train_source_loader:
            step += 1
            start_ts = time.time()
            model.train()
            
            images = images.cuda()
            labels = labels.cuda()

            # train
            optimizer.zero_grad()
            outputs = model(images)
            if args.net == '' or args.net == 'resnet50_fcn' or args.net == 'deeplabv3' or args.net == 'dl_mobilenet' or args.net == 'lraspp_mobilenet':
                outputs = outputs['out']  # rn50-FCN has outputs['out'] (pixel pred) and outputs['aux'] (pixel loss)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            # decrease lr
            if step == np.floor(args.steps * 0.75):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

        
            if step % args.log_interval == 0:
                log_info = OrderedDict({
                    'Train Step': step,
                    'Time/Image [s]': FormattedLogItem(time_meter.val / args.batch_size, '{:.3f}'),
                    'Batch domain': 'Source'
                })
                log_info.update({
                    'CE_2D Loss': FormattedLogItem(loss.item(), '{:.6f}')
                })

                log_str = get_log_str(args, log_info, title='Training Log')
                print(log_str)
                wandb.log(rm_format(log_info))
            
            if step % args.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for (images_val, labels_val) in val_loader:
                    #for (images_val, labels_val) in tqdm(val_loader):
                        images_val = images_val.cuda()
                        labels_val = labels_val.cuda()

                        outputs = model(images_val)
                        if args.net == '' or args.net == 'resnet50_fcn' or args.net == 'deeplabv3' or args.net == 'dl_mobilenet' or args.net == 'lraspp_mobilenet':
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

        for (images, labels) in train_target_loader:
            step += 1
            start_ts = time.time()
            model.train()
            
            images = images.cuda()
            labels = labels.cuda()

            # train
            optimizer.zero_grad()
            outputs = model(images)
            if args.net == '' or args.net == 'resnet50_fcn' or args.net == 'deeplabv3' or args.net == 'dl_mobilenet' or args.net == 'lraspp_mobilenet':
                outputs = outputs['out']  # rn50-FCN has outputs['out'] (pixel pred) and outputs['aux'] (pixel loss)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            # decrease lr
            if step == np.floor(args.steps * 0.75):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

        
            if step % args.log_interval == 0:
                log_info = OrderedDict({
                    'Train Step': step,
                    'Time/Image [s]': FormattedLogItem(time_meter.val / args.batch_size, '{:.3f}'),
                    'Batch domain': 'Target'
                })
                log_info.update({
                    'CE_2D Loss': FormattedLogItem(loss.item(), '{:.6f}')
                })

                log_str = get_log_str(args, log_info, title='Training Log')
                print(log_str)
                wandb.log(rm_format(log_info))
            
            if step % args.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for (images_val, labels_val) in val_loader:
                    #for (images_val, labels_val) in tqdm(val_loader):
                        images_val = images_val.cuda()
                        labels_val = labels_val.cuda()

                        outputs = model(images_val)
                        if args.net == '' or args.net == 'resnet50_fcn' or args.net == 'deeplabv3' or args.net == 'dl_mobilenet' or args.net == 'lraspp_mobilenet':
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
    wandb.init(name=args.expt_name, dir=args.save_dir,
               config=args, reinit=True, project=args.project, entity=args.entity)

    os.makedirs(args.save_dir, exist_ok=True)
    main(args, wandb)
    wandb.join()
    

# python main.py --steps=10001 --dataset=multi --source=real --target=sketch --backbone=expts/rot_pred/checkpoint.pth.tar --vat_tw=0 --expt_name=no_pretrain &
# python main.py --resume=expts/tmp_last/checkpoint.pth.tar --steps=10001 --dataset=multi --source=real --target=sketch --backbone=expts/rot_pred/checkpoint.pth.tar --vat_tw=0 --expt_name=run4 &