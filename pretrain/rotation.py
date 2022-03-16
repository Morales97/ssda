import os
import sys
from collections import OrderedDict

sys.path.append(os.path.abspath('..'))
#sys.path.insert(0, '/home/danmoral/PAC')
#sys.path.insert(0, '/Users/dani/Google Drive/My Drive/Uni/Master/EPFL/Thesis/Few Shot Domain Adaptation/repos/PAC_local')

import pdb

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large
from model.resnet import lraspp_mobilenetv3_large, Predictor
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

    s_loader = gtaLoader(image_path='../data/gta5/images_tiny', label_path='../data/gta5/labels', img_size=(360, 680), split="all_gta", rotation=True)
    t_loader = cityscapesLoader2(image_path='../data/cityscapes/leftImg8bit_tiny', label_path='../data/cityscapes/gtFine', img_size=(256, 512), split='train', rotation=True)
    
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
        model = mobilenet_v3_large(pretrained=True, dilated=True)   # dilated only removes last downsampling. for (224, 224) input, (14, 14) feature maps instead of (7,7) 
        model_layers = nn.Sequential(*list(model.children())[:-1])  # remove FC layers
        clas_head = Predictor(num_class=4, inc=960, temp=0.05, hidden=[])
        G = nn.Sequential(model_layers, clas_head)
        #pdb.set_trace()
    else:
        raise ValueError('Model cannot be recognized.')

    G.cuda()
    G.train()

    # prediction head has x10 LR
    # TODO try with same LR
    params = []
    for key, value in dict(G.named_parameters()).items():
        if value.requires_grad:
            if '1.fc' in key:   # '1.fc' corresponds to clas_head
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

        all_imgs = torch.cat((images_s, images_t), dim=0)
        all_labels = torch.cat((rot_labels_s, rot_labels_t), dim=0)

        pdb.set_trace()

        with autocast():
            all_imgs = torch.cat((im_data_s, im_data_t), dim=0)
            all_labels = torch.cat(
                (rot_labels_s, rot_labels_t), dim=0)
            all_feats = G(all_imgs)
            preds = F2(all_feats)
            loss = criterion(preds, all_labels)

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_g)
        scaler.step(optimizer_f)
        scaler.update()

        if step % args.log_interval == 0:
            log_info = OrderedDict({
                'Train Step': step,
                'Loss': FormattedLogItem(loss.item(), '{:.6f}'),
            })
            wandb.log(rm_format(log_info))
            log_str = get_log_str(args, log_info, title='Training Log')
            print(log_str)

        if step % args.save_interval == 0 and step > 0:
            with torch.no_grad():
                acc_s, acc_t = validate(
                    G, F2, source_loader, target_loader)
            log_info = OrderedDict({
                'Train Step': step,
                'Source Acc': FormattedLogItem(100. * acc_s, '{:.2f}'),
                'Target Acc': FormattedLogItem(100. * acc_t, '{:.2f}')
            })
            wandb.log(rm_format(log_info))
            log_str = get_log_str(args, log_info, title='Validation Log')
            print(log_str)
            G.train()
            F2.train()

            torch.save({
                'Train Step' : step,
                'G_state_dict' : G.state_dict(),
                'F2_state_dict' : F2.state_dict(),
                'optimizer_g_state_dict' : optimizer_g.state_dict(),
                'optimizer_f_state_dict' : optimizer_f.state_dict(),
            }, os.path.join(args.save_dir, 'checkpoint.pth.tar'))

            if step % (args.save_interval * args.ckpt_freq) == 0:
                shutil.copyfile(
                    os.path.join(args.save_dir, 'checkpoint.pth.tar'),
                    os.path.join(
                        args.save_dir, 'checkpoint_{}.pth.tar'.format(step)))

                # DM. save model as wandb artifact
                path_save_model = os.path.join(args.save_dir, 'G_state_dict.pth')
                torch.save(G.state_dict(), path_save_model)
                model_artifact = wandb.Artifact('model_{}'.format(step), type='model')
                model_artifact.add_file(path_save_model)
                wandb.log_artifact(model_artifact)


if __name__ == '__main__':
    args = parse_args()
    args.net = 'lraspp_mobilenet' # TODO DELETE
    os.makedirs(args.save_dir, exist_ok=True)

    #wandb.init(name=args.expt_name, dir=args.save_dir,
    #           config=args, reinit=True, project=args.project, entity=args.entity)
    wandb = None
    main(args, wandb)

    #wandb.join()


