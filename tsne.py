
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
from loader.loaders import get_loaders, get_loaders_pseudolabels
from utils.ioutils import parse_args
import pdb

def main(args):
    torch.set_num_threads(args.max_num_threads)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print('Seed: ', args.seed)
    

    args.batch_size_tl = 1
    # Load data
    if args.pseudolabel_folder is None:
        source_loader, target_loader, target_loader_unl, val_loader, _ = get_loaders(args)
    else:
        source_loader, target_loader, target_loader_unl, val_loader, _ = get_loaders_pseudolabels(args)
    
    # Init model and EMA
    model = get_model(args)
    model.cuda()
    model.train()
    '''
    ema_model = get_model(args)
    ema_model.cuda()
    ema_model.train()
    for param in ema_model.parameters():
        param.detach_()
    '''

    # To resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            #ema_model.load_state_dict(checkpoint['ema_state_dict'])
            print('*** Loading checkpoint from ', args.resume)

        else:
            raise Exception('No file found at {}'.format(args.resume))

    model.eval()
    ema_model.eval()

    for image, label in val_loader:
        out = model(image)
        feat = out['feat']
        pdb.set_trace()


if __name__ == '__main__':
    print('Running tsne.py')
    args = parse_args()
    main(args)

# python tsne.py --resume=expts/tmp_last/checkpoint_abl_noPCmix_p2_3.pth.tar