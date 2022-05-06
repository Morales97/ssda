from __future__ import absolute_import

import torchvision
import pdb
import torch.nn as nn
import torch
from torchsummary import summary

#from model.fcn import fcn8s, fcn_resnet50, fcn_resnet50_densecl
from model.deeplabv3 import deeplabv3_rn50
from model.deeplabv2 import deeplabv2_rn101
from model.deeplabv2old import deeplabv2_rn101 as deeplabv2_rn101_old
from model.lraspp import lraspp_mobilenetv3_large
from model.lraspp_contrast import lraspp_mobilenet_v3_large_contrast

def get_model(args):
    if args.net == 'deeplabv3_rn50':
        model = deeplabv3_rn50(args.pre_trained, args.pre_trained_backbone, args.custom_pretrain, args.dsbn)
    elif args.net == 'deeplabv2_rn101':
        model = deeplabv2_rn101(args.pre_trained, args.pre_trained_backbone, args.custom_pretrain)
    elif args.net == 'deeplabv2_rn101_old':
        model = deeplabv2_rn101_old(args.pre_trained, args.pre_trained_backbone, args.custom_pretrain)
    #elif args.net == 'dl_mobilenet':
    #    model = deeplabv3_mobilenetv3_large(args.pre_trained, args.pre_trained_backbone)
    elif args.net == 'lraspp_mobilenet':
        model = lraspp_mobilenetv3_large(args.pre_trained, args.pre_trained_backbone, args.custom_pretrain)
    elif args.net == 'lraspp_mobilenet_contrast':
        model = lraspp_mobilenet_v3_large_contrast(args.pre_trained, args.pre_trained_backbone, args.custom_pretrain)

    return model



