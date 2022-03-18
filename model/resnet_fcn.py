from torch import nn
import pdb
import torch
from torchsummary import summary
from torchvision.models import resnet
import torch.nn.functional as F

'''
modified from original torch's implementation
https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
'''

class FCN(nn.Module):

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


# NOTE auxiliary classifier removed -- compared to torch's implementation
def _fcn_resnet(backbone: resnet.ResNet, num_classes: int) -> FCN:
    classifier = FCNHead(2048, num_classes)
    return FCN(backbone, classifier)


def fcn_resnet50(num_classes=19):
    # pretrained weights on coco are available - check torch
    backbone = nn.Sequential(*list(
        resnet.resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True]).children())[:-2])
    model = _fcn_resnet(backbone, num_classes)
    return model

def fcn_resnet50_densecl(num_classes=19):
    # pretrained model from https://github.com/WXinlong/DenseCL 
    densecl_pretrained = torch.load('model/pretrained/densecl_r50_imagenet_200ep.pth')
    rn50 = resnet.resnet50(replace_stride_with_dilation=[False, True, True])
    sd = densecl_pretrained['state_dict']
    rn50_sd = rn50.state_dict()

    # NOTE sd is missing (fc) keys, add them
    for key, param in rn50_sd.items():
        if key not in sd.keys():
            sd[key] = param

    rn50.load_state_dict(sd)

    backbone = nn.Sequential(*list(rn50.children())[:-2])
    model = _fcn_resnet(backbone, num_classes)
    return model


def fcn_resnet34(num_classes=19):
    # pretrained weights on coco are available - check torch
    # TODO: does not work bc resnet34 is built of BasicBlock, while resnet50 is built of Bottlenecks.
    # Only Bottlenecks support dilation in torch's implementation
    backbone = nn.Sequential(*list(
        resnet.resnet34(pretrained=False, replace_stride_with_dilation=[False, True, True]).children())[:-2])
    model = _fcn_resnet(backbone, num_classes)
    return model

def fcn_resnet18(num_classes=19):
    # pretrained weights on coco are available - check torch
    backbone = resnet.resnet18(pretrained=False, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes)
    return model



#fcn18 = fcn_resnet18()
#fcn34 = fcn_resnet34()
#fcn50 = fcn_resnet50()
#densecl_rn50 = fcn_resnet50_densecl()

#pdb.set_trace()
