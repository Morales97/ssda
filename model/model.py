from __future__ import absolute_import

import torchvision
import pdb
import torch.nn as nn
import torch
from torchsummary import summary

from model.deeplabv3 import deeplabv3_rn50
from model.deeplabv2 import deeplabv2_rn101

def get_model(args):
    if args.net == 'deeplabv3_rn50':
        model = deeplabv3_rn50(args.pre_trained, args.pre_trained_backbone, args.custom_pretrain, args.dsbn)
    elif args.net == 'deeplabv2_rn101':
        model = deeplabv2_rn101(args.pre_trained, args.pre_trained_backbone, args.custom_pretrain, args.pc_memory, args.alonso_contrast)

    return model



