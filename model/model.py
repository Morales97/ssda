from __future__ import absolute_import

import torchvision
import pdb
import torch.nn as nn
import torch
from torchsummary import summary

from model.fcn import fcn8s, fcn_resnet50, fcn_resnet50_densecl
from model.deeplab import deeplabv3_resnet50_maskContrast, deeplabv3_rn50, deeplabv3_mobilenetv3_large
from model.lraspp import lraspp_mobilenetv3_large

def get_model(args):
    if args.net == 'fcn_resnet50':
        model = fcn_resnet50(args.pre_trained)
    elif args.net == 'denseCL_fcn_rn50':
        model = fcn_resnet50_densecl()
    elif args.net == 'fcn8':
        model = fcn8s()
    elif args.net == 'deeplabv3_rn50':
        model = deeplabv3_rn50(args.pre_trained, args.pre_trained_backbone)
    elif args.net == 'deeplabv3_mask_pt':
        model = deeplabv3_resnet50_maskContrast(model_path=args.custom_pretrain_path)
    elif args.net == 'dl_mobilenet':
        model = deeplabv3_mobilenetv3_large(args.pre_trained, args.pre_trained_backbone)
    elif args.net == 'lraspp_mobilenet':
        model = lraspp_mobilenetv3_large(args.pre_trained, args.pre_trained_backbone, args.custom_pretrain_path)

    return model



