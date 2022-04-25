
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

affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

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
    def __init__(self, block, layers, num_classes, pixel_contrast=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_classes= num_classes
        self.pixel_contrast = pixel_contrast

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
        if self.pixel_contrast:
            self.projection = nn.Sequential(
                            nn.Conv2d(2048, 2048, 1, bias=False),
                            nn.BatchNorm2d(2048),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(2048, 256, 1, bias=False)
                        )


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


    def forward(self, x, return_features=False):
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
        result['out'] = x

        if self.pixel_contrast:
            proj = self.projection(features)
            proj = F.normalize(proj, p=2, dim=1)  
            proj = F.interpolate(proj, size=input_shape, mode="bilinear", align_corners=False)
            result["proj"] = proj

        if return_features:
            return result, features
        else:
            return result


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


def deeplabv2_rn101(pretrained=False, pretrained_backbone=True, custom_pretrain_path=None, pixel_contrast=False, num_classes=19):
    if pretrained:
        raise Exception('pretrained DeepLabv2 + ResNet-101 is not available')

    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, pixel_contrast)
    
    if custom_pretrain_path is not None:
        print('Loading model from %s' % custom_pretrain_path)
        maskContrast_pretrained = torch.load(custom_pretrain_path, map_location=torch.device('cpu'))
        sd = maskContrast_pretrained['model']

        # Create a new state_dict
        new_state_dict = {}
        for key, param in sd.items():
            if 'module.model_q.' in key:
                if 'backbone' in key:
                    new_state_dict[key[15:]] = param  # remove the 'module.model_q.' part
                    print(key)
                elif 'head' in key:
                    new_state_dict['layer5' + key[19:]] = param
                    print(key)

        model.load_state_dict(new_state_dict, strict=False) 
        pdb.set_trace()
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
    model = deeplabv2_rn101(pixel_contrast=False, custom_pretrain_path='model/pretrained/ckpt_mask_v2.tar')
    pdb.set_trace()
