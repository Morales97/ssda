from __future__ import absolute_import
from typing import List, Optional, Dict

import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
import pdb
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from model.dsbn import resnet_dsbn
#from dsbn import resnet_dsbn
from collections import OrderedDict
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, to_tensor
import copy
import pickle

'''
torch's deeplab from
https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py 
'''


__all__ = [
    "DeepLabV3",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
]


model_urls = {
    "deeplabv3_resnet50_coco": "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
    "deeplabv3_resnet101_coco": "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
    "deeplabv3_mobilenet_v3_large_coco": "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
}


class DeepLabV3_custom(nn.Module):
    '''
    a custom deeplabv3 model that combines all elements:
        - pixel contrast cross-image
        - Alonso's PC
        - DSBN
    '''
    def __init__(self, backbone: nn.Module, in_channels, num_classes, dim_embed=256, is_dsbn=False) -> None:
        super().__init__()
        self.is_dsbn = is_dsbn
        self.memory_size = 1000 #5000 # TODO un-hardcode
        self.pixel_update_freq = 10 # TODO un-hardcode
        self.ignore_label = 250

        self.backbone = backbone
        self.aspp = ASPP(in_channels, [12, 24, 36])
        self.decoder1 = nn.Sequential(
                            nn.Conv2d(256, 256, 3, padding=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                        )
        self.decoder2 = nn.Conv2d(256, num_classes, 1)
        
        # for pixel contrast
        self.projection_pc = nn.Sequential(
                            nn.Conv2d(256, 256, 1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, dim_embed, 1, bias=False)
                        )
        
        # segment_queue is the "region" memory
        self.register_buffer("segment_queue", torch.randn(num_classes, self.memory_size, dim_embed))
        self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

        # pixel_queue is the "pixel" memory
        self.register_buffer("pixel_queue", torch.randn(num_classes, self.memory_size, dim_embed))
        self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
        self.register_buffer("pixel_queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        
        # for Alonso's PC
        dim_in = 256
        feat_dim = 256
        self.projection_head = nn.Sequential(
                                    nn.Conv2d(dim_in, feat_dim, 1, bias=False), # difference to original Alonso: nn.Linear(dim_in, feat_dim)
                                    nn.BatchNorm2d(feat_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
                                )
        self.prediction_head = nn.Sequential(
                                    nn.Conv2d(feat_dim, feat_dim, 1, bias=False),
                                    nn.BatchNorm2d(feat_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
                                )
        
        for class_c in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)
        
        print('Using custom DeepLabV3 model')


    def forward(self, x: Tensor, domain=None) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        if self.is_dsbn:
            features = self.backbone(x, domain)
        else:
            features = self.backbone(x)

        result = OrderedDict()  
        aspp_f = self.aspp(features["out"])     # aspp_f is input to projection_pc head
        x_f = self.decoder1(aspp_f)             # x_f is input to projection head
        x = self.decoder2(x_f)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        proj = self.projection_head(x_f)        # proj and pred heads are not upsampled -> CL occurs in the lower resolution, they sample down the labels and images
        pred = self.prediction_head(proj)

        proj_pc = self.projection_pc(aspp_f)
        proj_pc = F.normalize(proj_pc, p=2, dim=1)  #need to normalize the projection
        proj_pc = F.interpolate(proj_pc, size=input_shape, mode="bilinear", align_corners=False)

        result["out"] = x
        result["feat"] = x_f
        result["proj"] = proj
        result["pred"] = pred
        result["proj_pc"] = proj_pc

        return result

    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        # I think this selects one in every network_stride values. it is a way to downsample the label. Not necessery if I have downsampled it before
        #labels = labels[:, ::self.network_stride, ::self.network_stride]    
        # TODO assert that commenting this line is fine

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_label]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.memory_size:
                    self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + K) % self.memory_size
                
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


def _deeplabv3_resnet50(
    pretrained_backbone: bool, 
    dsbn: bool,
    in_channels = 2048,
    num_classes = 19
) -> DeepLabV3_custom:
    
    return_layers = {"layer4": "out"}
    if pretrained_backbone: print('Loading backbone with ImageNet weights')

    if dsbn:
        backbone = resnet_dsbn.resnet50dsbn(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
        backbone = resnet_dsbn.IntermediateLayerGetterDSBN(backbone, return_layers=return_layers)
        return DeepLabV3_custom(backbone, 2048, num_classes, is_dsbn=True)
    else:
        backbone = resnet.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        return DeepLabV3_custom(backbone, 2048, num_classes, 256)

def deeplabv3_rn50(pretrained=False, pretrained_backbone=True, custom_pretrain=None, dsbn=False):
    if custom_pretrain is not None:
        pretrained_backbone=False

    model = _deeplabv3_resnet50(pretrained_backbone, dsbn=dsbn)   

    if custom_pretrain == 'denseCL':
        return _load_denseCL(model)

    if custom_pretrain == 'pixpro':
        return _load_pixpro(model)

    if custom_pretrain is not None:
        print('Loading model from %s' % custom_pretrain)
        state_dict = torch.load(custom_pretrain)['model']

        # For DeepLabContrast2, the state dict is different. TODO change this when merging models
        # Create a new state_dict
        new_state_dict = copy.deepcopy(model.state_dict())
        for key, param in state_dict.items():
            if 'model_q.' in key:
                if 'backbone' in key:
                    new_state_dict[key[8:]] = param  # remove the 'model_q.' part
                elif 'decoder.0' in key:
                    new_state_dict['aspp' + key[17:]] = param
                # TODO could also upload last decoder layer (very few params)

        model.load_state_dict(new_state_dict)
        return model

    return model


def _load_denseCL(model, path='model/pretrained/densecl_r50_imagenet_200ep.pth'):
    pt_sd = torch.load(path)['state_dict']

    new_state_dict = copy.deepcopy(model.state_dict())
    for key, param in pt_sd.items():
        new_state_dict['backbone.' + key] = param
            
    model.load_state_dict(new_state_dict)
    print('Loading model pretrained densly on ImageNet with DenseCL')
    return model


def _load_pixpro(model, path='pretrained/pixpro_base_r50_400ep.pth'):
    pt_sd = torch.load(path)['model']

    new_state_dict = copy.deepcopy(model.state_dict())
    for key, param in pt_sd.items():
        if 'module.encoder' in key and not 'encoder_k' in key:
            new_state_dict['backbone' + key[14:]] = param 
            
    print('Loading model pretrained densly on ImageNet with PixPro')
    model.load_state_dict(new_state_dict)
    return model


if __name__ == '__main__':
    #pt_sd = np.load('model/pretrained/resnet50_detcon_b_imagenet_1k.npy', allow_pickle=True).item() # NOTE need .item() to get dat from np.object
    # the keys are saved in a completely different format, not suitable to load model...
    model = deeplabv3_rn50(dsbn=False)
    for i, elem in enumerate(model.parameters()):
        print(i)    
    pdb.set_trace()
