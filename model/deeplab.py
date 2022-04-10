from typing import List, Optional, Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import pdb
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter

from collections import OrderedDict


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


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


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


def _deeplabv3_resnet(
    backbone: resnet.ResNet,
    num_classes: int,
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, None)


def deeplabv3_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    # DM: modified to remove aux classifier

    backbone = resnet.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes)

    return model



def deeplabv3_rn50(pretrained=False, pretrained_backbone=True, custom_pretrain_path=None):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', 
        pretrained=pretrained, 
        num_classes=19, 
        pretrained_backbone=pretrained_backbone
    )
    return model



def deeplabv3_mobilenetv3_large(pretrained=False, pretrained_backbone=True):
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        pretrained=pretrained,
        num_classes=19,
        pretrained_backbone=pretrained_backbone
    )
    return model

def deeplabv3_resnet50_maskContrast(num_classes=19, model_path=None):
    # Load a pretrained DeepLabV3_rn50 on PASCAL VOC.
    # pretrained model from https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation (on PASCAL VOC unsupervised saliency)
    
    assert model_path is not None
    # for PASCAL pretrained model (sup):    model/pretrained/VOCSegmentation_supervised_saliency_model.pth.tar
    # for PASCAL pretrained model (unsup):  model/pretrained/VOCSegmentation_unsupervised_saliency_model.pth.tar 
    # for CS pretrained model at epoch 39:  model/pretrained/checkpoint_39_mask_dlrn50.pth.tar
    # for CS + GTA pretrained at epoch 32:  model/pretrained/checkpoint_mask_dlrn50_CS_GTA.pth.tar

    print('Loading model from %s' % model_path)
    maskContrast_pretrained = torch.load(model_path, map_location=torch.device('cpu'))
    model = deeplabv3_resnet50(num_classes=num_classes)   
    sd = maskContrast_pretrained['model']

    # Create a new state_dict
    new_state_dict = {}
    for key, param in sd.items():
        if 'module.model_q.' in key:
            if 'backbone' in key:
                new_state_dict[key[15:]] = param  # remove the 'module.model_q.' part
            elif 'decoder' in key:
                new_state_dict['classifier' + key[22:]] = param

    model.load_state_dict(new_state_dict, strict=False) 
    return model

'''
model = deeplabv3_resnet50_maskContrast(model_path='model/pretrained/VOCSegmentation_unsupervised_saliency_model.pth.tar')
pdb.set_trace()
'''