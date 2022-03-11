import os
import random
import shutil
from collections import OrderedDict
import time

#import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.resnet import resnet50_FCN
#from utils.eval import test
from utils.ioutils import FormattedLogItem
from utils.ioutils import gen_unique_name
from utils.ioutils import get_log_str
from utils.ioutils import parse_args
from utils.ioutils import rm_format
from loader.cityscapes_loader import cityscapesLoader
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

    # TODO: rn loaders don't use transforms or augmentations. Probably should be using some
    t_loader = cityscapesLoader(image_path='data/cityscapes/leftImg8bit_tiny', label_path='data/cityscapes/gtFine', img_size=(256, 512), split='train')
    v_loader = cityscapesLoader(image_path='data/cityscapes/leftImg8bit_tiny', label_path='data/cityscapes/gtFine', img_size=(256, 512), split='val')

    train_loader = DataLoader(
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
    model = resnet50_FCN(args.pre_trained)

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
            optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step'] + 1
            print('Resuming from train step {}'.format(start_step))
        else:
            raise Exception('No file found at {}'.format(args.resume))

    # Custom loss function. Ignores index 250
    loss_fn = cross_entropy2d   

    best_acc = 0 
    step = start_step
    time_meter = averageMeter()
    val_loss_meter = averageMeter()

    while step <= args.steps:
        for (images, labels) in train_loader:
            step += 1
            start_ts = time.time()
            model.train()
            
            images = images.cuda()
            labels = labels.cuda()

            # train
            optimizer.zero_grad()
            outputs = model(images)['out']  # rn50-FCN has outputs['out'] (pixel pred) and outputs['aux'] (pixel loss)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)
            # decrease lr
            if step == args.steps/2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

        
            if step % args.log_interval == 0:
                log_info = OrderedDict({
                    'Train Step': step,
                    'Time/Image': time_meter.val / args.batch_size
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
                    pdb.set_trace()
                    for (images_val, labels_val) in val_loader:
                    #for (images_val, labels_val) in tqdm(val_loader):
                        images_val = images_val.cuda()
                        labels_val = labels_val.cuda()
                        pdb.set_trace()
                        outputs = model(images_val)['out']
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                log_info = OrderedDict({
                    'Train Step': step,
                    'Validation loss': val_loss_meter.avg
                })

                log_str = get_log_str(args, log_info, title='Validation Log')
                print(log_str)
                wandb.log(rm_format(log_info))
                
                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    # TODO add to wandb logger

                for k, v in class_iou.items():
                    print(k, v)
                    # TODO add to wandb logger

                val_loss_meter.reset()
                running_metrics_val.reset()


            if step % args.save_interval == 0 and step > 0:
                if args.save_model:
                    torch.save({
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'step' : step,
                    }, os.path.join(args.save_dir, 'checkpoint.pth.tar'))

                # TODO save best model
                '''
                if acc_val > best_acc:
                    if args.save_model:
                        shutil.copyfile(
                            os.path.join(args.save_dir, 'checkpoint.pth.tar'),
                            os.path.join(args.save_dir, 'model-best.pth.tar'))
                    best_acc = acc_val
                    best_acc_test = acc_test
                
                    # DM. save model as wandb artifact
                    model_artifact = wandb.Artifact('best_model_{}'.format(step), type='model')
                    model_artifact.add_file(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
                    wandb.log_artifact(model_artifact)
                '''


if __name__ == '__main__':
    args = parse_args()
    main(args, wandb)

    # W&B logging setup
    #wandb = WandbWrapper(debug=~args.use_wandb)
    if not args.expt_name:
        args.expt_name = gen_unique_name()
    if args.project == '':
        args.project = 'seg_test'
        entity = 'morales97'
    wandb.init(name=args.expt_name, dir=args.save_dir,
               config=args, reinit=True, project=args.project, entity=entity)

    os.makedirs(args.save_dir, exist_ok=True)
    main(args, wandb)
    wandb.join()
    

# python main.py --steps=10001 --dataset=multi --source=real --target=sketch --backbone=expts/rot_pred/checkpoint.pth.tar --vat_tw=0 --expt_name=no_pretrain &
# python main.py --resume=expts/tmp_last/checkpoint.pth.tar --steps=10001 --dataset=multi --source=real --target=sketch --backbone=expts/rot_pred/checkpoint.pth.tar --vat_tw=0 --expt_name=run4 &