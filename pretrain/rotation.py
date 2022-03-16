import os
import sys
from collections import OrderedDict
import time

#sys.path.append(os.path.abspath('..'))
sys.path.insert(0, '/home/danmoral/seg_test')
#sys.path.insert(0, '/Users/dani/Google Drive/My Drive/Uni/Master/EPFL/Thesis/Few Shot Domain Adaptation/repos/PAC_local')

import pdb

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large
from model.resnet import lraspp_mobilenetv3_large, Predictor, RotationPred
#from utils.eval import test
from utils.ioutils import FormattedLogItem
from utils.ioutils import gen_unique_name
from utils.ioutils import get_log_str
from utils.ioutils import parse_args
from utils.ioutils import rm_format
from loader.cityscapes_loader2 import cityscapesLoader2
from loader.gta_loader import gtaLoader
from loss.loss import cross_entropy2d
from evaluation.metrics import averageMeter, runningScore
import wandb
import tqdm


def validate(G, F2, loader_s, loader_t):
    G.eval()
    F2.eval()
    acc_s = AverageMeter()
    acc_t = AverageMeter()
    accs = [acc_s, acc_t]
    loaders = [loader_s, loader_t]

    '''
    torch.backends.cudnn.benchmark = False
    for lidx, loader in enumerate(loaders):
        for i, data in enumerate(loader):
            imgs = data[0].reshape((-1,) + data[0].shape[2:]).cuda()
            rot_labels = data[2].reshape((-1,) + data[2].shape[2:]).cuda()
            preds = F2(G(imgs))
            preds = preds.argmax(dim=1)
            accs[lidx].update(
                (preds == rot_labels).sum()/float(len(imgs)), len(imgs))
    torch.backends.cudnn.benchmark = True
    '''
    return 0, 0
    return acc_s.avg, acc_t.avg

def main(args, wandb):
    torch.set_num_threads(args.max_num_threads)

    torch.manual_seed(args.seed)

    s_loader = gtaLoader(image_path='data/gta5/images_tiny', label_path='data/gta5/labels', img_size=(360, 680), split="all_gta", rotation=True)
    t_loader = cityscapesLoader2(image_path='data/cityscapes/leftImg8bit_tiny', label_path='data/cityscapes/gtFine', img_size=(256, 512), split='train', rotation=True)
    
    source_loader = DataLoader(
        s_loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    target_loader = DataLoader(
        t_loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )    

    # Model
    if args.net == 'lraspp_mobilenet':
        backbone = mobilenet_v3_large(pretrained=True, dilated=True)   # dilated only removes last downsampling. for (224, 224) input, (14, 14) feature maps instead of (7,7) 
        backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove FC layers
        clas_head = Predictor(num_class=4, inc=960, temp=0.05, hidden=[])
        model = RotationPred(backbone, clas_head)
        #pdb.set_trace()
    else:
        raise ValueError('Model cannot be recognized.')

    model.cuda()
    model.train()

    # prediction head has x10 LR
    # TODO try with same LR
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'classifier' in key: 
                params += [{'params': [value], 'lr': 10*args.lr,
                            'weight_decay': args.wd}]
            else:
                params += [{'params': [value], 'lr': args.lr,
                            'weight_decay': args.wd}]


    # The classifiers have 10 times the learning rate of the backbone
    optimizer = optim.SGD(params, momentum=0.9,
                            weight_decay=args.wd, nesterov=True)

    start_step = 0

    '''
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            G.load_state_dict(checkpoint['G_state_dict'])
            F2.load_state_dict(checkpoint['F2_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_f.load_state_dict(checkpoint['optimizer_f_state_dict'])
            start_step = checkpoint['Train Step'] + 1
            print('Resuming from train step {}'.format(start_step))
        else:
            raise Exception('No file found at {}'.format(args.resume))
    '''


    criterion = nn.CrossEntropyLoss()
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)

    for step in range(start_step, args.steps):
        start_ts = time.time()

        if step % len(target_loader) == 0:
            data_iter_t = iter(target_loader)
        if step % len(source_loader) == 0:
            data_iter_s = iter(source_loader)
        
        images_s, rot_lbl_s = next(data_iter_t)
        images_t, rot_lbl_t = next(data_iter_s)

        images_s = images_s.flatten(end_dim=1).cuda()      # (B, 4, C, H, W) -> (B·4, C, H, W)
        images_t = images_t.flatten(end_dim=1).cuda()
        rot_lbl_s = rot_lbl_s.flatten().cuda()             # (B, 4) -> (B·4)
        rot_lbl_s = rot_lbl_s.flatten().cuda()

        pdb.set_trace()
        preds_s = model(images_s)
        preds_t = model(images_t)
        loss = 0
        loss += criterion(preds_s, rot_lbl_s)
        loss += criterion(preds_t, rot_lbl_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - start_ts)

        if step % args.log_interval == 0:
            log_info = OrderedDict({
                'Train Step': step,
                'Time/Image [s]': FormattedLogItem(time_meter.val / args.batch_size*2, '{:.3f}')
            })
            log_info.update({
                'Rotation CE Loss': FormattedLogItem(loss.item(), '{:.6f}')
            })

            log_str = get_log_str(args, log_info, title='Training Log')
            print(log_str)
            wandb.log(rm_format(log_info))

 
        if step % args.save_interval == 0:
            if args.save_model:
                torch.save({
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'step' : step,
                }, os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            
            # DM. save model as wandb artifact
            model_artifact = wandb.Artifact('checkpoint_{}'.format(step), type='model')
            model_artifact.add_file(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            wandb.log_artifact(model_artifact)


if __name__ == '__main__':
    args = parse_args()
    args.save_dir = 'expts_rot/tmp_last'
    os.makedirs(args.save_dir, exist_ok=True)

    #wandb.init(name=args.expt_name, dir=args.save_dir,
    #           config=args, reinit=True, project=args.project, entity=args.entity)
    wandb = None
    main(args, wandb)

    #wandb.join()


