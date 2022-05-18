
"""
This is the implementation of DeepLabv2 without multi-scale inputs. This implementation uses ResNet-101 as backbone.
This deeplab is used with imagenet pretraining to match the current pytorch implementation that provides these weights.
This implementation follows the new implementation of Resnet bottleneck module where the stride is performed in the 3x3 conv.

Code taken from https://github.com/WilhelmT/ClassMix, slightly modified
"""

import torch
import torch.nn as nn
from torch.utils import model_zoo
import numpy as np
from torch.nn import functional as F
import pdb
from collections import OrderedDict
import copy 

affine_par = True



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, pc_memory=False, alonso_contrast=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_classes= num_classes
        self.dim_embed = 256
        self.memory_size = 1000 #5000
        self.pixel_update_freq = 10
        self.ignore_label = 250

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = _ASPP(2048, num_classes, [6,12,18,24])
        
        # for Pixel Contrast
        self.projection_pc = nn.Sequential(
                        nn.Conv2d(2048, 2048, 1, bias=False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(2048, self.dim_embed, 1, bias=False)
                    )

        if pc_memory:
            # segment_queue is the "region" memory
            self.register_buffer("segment_queue", torch.randn(num_classes, self.memory_size, self.dim_embed))
            self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
            self.register_buffer("segment_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

            # pixel_queue is the "pixel" memory
            self.register_buffer("pixel_queue", torch.randn(num_classes, self.memory_size, self.dim_embed))
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
            self.register_buffer("pixel_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

        if alonso_contrast:
            # for Alonso's PC
            dim_in = 2048
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
            
        print('Using custom DeepLabV2 model')


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)


    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)

        result = OrderedDict()
        x = self.layer5(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        proj = self.projection_head(features)        # proj and pred heads are not upsampled -> CL occurs in the lower resolution, they sample down the labels and images
        pred = self.prediction_head(proj)

        proj_pc = self.projection_pc(features)
        proj_pc = F.normalize(proj_pc, p=2, dim=1)  
        proj_pc = F.interpolate(proj_pc, size=input_shape, mode="bilinear", align_corners=False)

        result["out"] = x
        result["feat"] = features
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

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        b.append(self.layer5)
        #b.append(self.projection_head)
        b.append(self.prediction_head)

        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k


    def optim_parameters(self, args):
        # TODO: change names
        return [{'params': self.get_1x_lr_params(), 'lr': args.learning_rate}]


def deeplabv2_rn101(pretrained=False, pretrained_backbone=True, custom_pretrain_path=None, pc_memory=False, alonso_contrast=None, num_classes=19):
    if pretrained:
        raise Exception('pretrained DeepLabv2 + ResNet-101 is not available')

    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, pc_memory, alonso_contrast)
    
    if custom_pretrain_path is not None:
        print('Loading model from %s' % custom_pretrain_path)
        maskContrast_pretrained = torch.load(custom_pretrain_path)
        sd = maskContrast_pretrained['model']

        # Create a new state_dict
        new_state_dict = copy.deepcopy(model.state_dict())
        for key, param in sd.items():
            if 'model_q.' in key:
                if 'backbone' in key:
                    new_state_dict[key[17:]] = param  # remove the 'module.model_q.' part
                # NOTE do not load classification head: v2 decoder is 2048 -> C, and C changes to num_classes. cannot reuse

        model.load_state_dict(new_state_dict)
        return model

    if pretrained_backbone:
        print('*** Loading ImageNet weights to DeepLabv2 + ResNet-101 ***')
        sd_imagenet = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')  # ImageNet pretrained rn101

        new_dict = {}
        for k, v in sd_imagenet.items():
            if not k.startswith('fc'):
                new_dict[k] = v
        model.load_state_dict(new_dict, strict=False)
    
        '''
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in sd_imagenet and param.size() == sd_imagenet[name].size():
                new_params[name].copy_(sd_imagenet[name])
                print(name)
        model.load_state_dict(new_params)
        '''
    return model


if __name__ == '__main__':

    model = deeplabv2_rn101(custom_pretrain_path='model/pretrained/ckpt_mask_v2.tar')
    pdb.set_trace()
