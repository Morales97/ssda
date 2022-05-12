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
from loader.loaders import get_loaders, get_loaders_pseudolabels
from evaluation.metrics import averageMeter, runningScore
from utils.lab_color import lab_transform
import wandb
from torch_ema import ExponentialMovingAverage # https://github.com/fadel/pytorch_ema 
from utils.class_balance import get_class_weights

from torchvision.utils import save_image
import pdb


def main(args):
    torch.set_num_threads(args.max_num_threads)
    print('*** Running generate_pseudolabels.py ...')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print('Seed: ', args.seed)

    # Load model
    model = get_model(args)
    model.cuda()
    model.train()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    ema.to(torch.device('cuda:' +  str(torch.cuda.current_device())))

    # Load trained model
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'ema_state_dict' in checkpoint.keys():
                ema.load_state_dict(checkpoint['ema_state_dict'])
        else:
            raise Exception('No file found at {}'.format(args.resume))
    else:
        raise Exception('No model found to generate pseudolabels with')

    pseudo_lbl_path = generate_pseudolabels(args, model, ema)

if __name__ == '__main__':
    args = parse_args()
    main(args)

# python generate_pseudolabels.py --seed=3 --size=small --expt_name=KL_pc_40k --net=deeplabv2_rn101 --resume=model/pretrained/model_40k_KL_pc.tar