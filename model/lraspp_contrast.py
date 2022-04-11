# Torch's source code


from collections import OrderedDict
from typing import Any, Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
#from torchvision.utils import _log_api_usage_once
from torchvision.models import mobilenetv3
from torchvision.models._utils import IntermediateLayerGetter
#from torchvision.models.segmentation._utils import _load_weights

import pdb

__all__ = ["LRASPP", "lraspp_mobilenet_v3_large"]


model_urls = {
    "lraspp_mobilenet_v3_large_coco": "https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
}


class LRASPP_Contrast(nn.Module):
    """
    DM: modified to include projection head for Pixel Contrast
    """

    def __init__(
        self, backbone: nn.Module, low_channels: int, high_channels: int, dim_embed: int, num_classes: int, inter_channels: int = 128
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)
        
        # NOTE this is an improvised pojection head for the LRASPP arch. Try both? In DeepLab, follow paper
        #self.projection = LRASPPHead(low_channels, high_channels, dim_embed, high_channels)
        self.projection = nn.Sequential(
                            LRASPPHead(low_channels, high_channels, dim_embed, high_channels),
                            nn.BatchNorm2d(dim_embed),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(dim_embed, dim_embed, 1, bias=False)
                        )

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)

        result = OrderedDict()
        result['out'] = out

        proj = self.projection(features)
        proj = F.normalize(proj, p=2, dim=1)  #need to normalize the projection
        proj = F.interpolate(proj, size=input.shape[-2:], mode="bilinear", align_corners=False)
        result['proj'] = proj

        return result


class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


def _lraspp_mobilenetv3(backbone: mobilenetv3.MobileNetV3, dim_embed: int, num_classes: int) -> LRASPP_Contrast:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels

    #pdb.set_trace()
    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})

    return LRASPP_Contrast(backbone, low_channels, high_channels, dim_embed, num_classes)


def lraspp_mobilenet_v3_large_contrast(
    pretrained: bool = False,
    progress: bool = True,
    dim_embed: int = 128,
    num_classes: int = 19,
    pretrained_backbone: bool = True,
    **kwargs: Any,
) -> LRASPP_Contrast:
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")
    if pretrained:
        pretrained_backbone = False

    backbone = mobilenetv3.mobilenet_v3_large(pretrained=pretrained_backbone, dilated=True)
    model = _lraspp_mobilenetv3(backbone, dim_embed, num_classes)

    if pretrained:
        # load COCO's weights
        model_coco = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)

        sd = model.state_dict()
        sd_coco = model_coco.state_dict()
        for k, v in sd_coco.items():
            if not (k.startswith('classifier.low_classifier') or k.startswith('classifier.high_classifier') or k.startswith('projection')):
                # Copy parameters in all common layers
                sd[k] = v

        model.load_state_dict(sd)
        return model

    if custom_pretrain_path is not None:
        # load pretrained backbone
        checkpoint = torch.load(custom_pretrain_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print('Loading model from ' + custom_pretrain_path)
        
        if 'model' in checkpoint.keys():
            # MaskContrast pretrain with head embedding dim = 32
            # Replace last layer of head with a new layer of output_dim=n_class
            state_dict = checkpoint['model']

            new_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model_q.backbone'):
                    new_dict[k.rsplit('model_q.')[1]] = v
                if k.startswith('model_q.decoder') and 'low' not in k and 'high' not in k:
                    new_dict['classifier' + k.rsplit('model_q.decoder')[1]] = v
        
        # copy matching keys of state dict -- all but for LRASPP head
        model.cuda()
        model.load_state_dict(new_dict, strict=False)

    return model

'''
def lraspp_mobilenetv3_large(pretrained=False, pretrained_backbone=True, custom_pretrain_path=None):
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        pretrained=False,
        num_classes=19,
        pretrained_backbone=pretrained_backbone
    )

    if pretrained:
        # load COCO's weights
        model_coco = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)

        sd = model.state_dict()
        sd_coco = model_coco.state_dict()
        for k, v in sd_coco.items():
            if not (k.startswith('classifier.low_classifier') or k.startswith('classifier.high_classifier')):
                # Copy all parameters but for linear classifier, which has different number of classes
                sd[k] = v

        model.load_state_dict(sd)
        return model

    if custom_pretrain_path is not None:
        # load pretrained backbone
        checkpoint = torch.load(custom_pretrain_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print('Loading model from ' + custom_pretrain_path)

        if 'model_state_dict' in checkpoint.keys():
            # Rotations pretrain (?)
            state_dict = checkpoint['model_state_dict']

            new_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.0'):
                    new_key = k.rsplit('.0')[0] + k.rsplit('.0')[1]     # remove '.0'
                    new_dict[new_key] = v 
        
        elif 'model' in checkpoint.keys():
            # MaskContrast pretrain with head embedding dim = 32
            # Replace last layer of head with a new layer of output_dim=n_class
            state_dict = checkpoint['model']

            new_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model_q.backbone'):
                    new_dict[k.rsplit('model_q.')[1]] = v
                if k.startswith('model_q.decoder') and 'low' not in k and 'high' not in k:
                    new_dict['classifier' + k.rsplit('model_q.decoder')[1]] = v
        
        # copy matching keys of state dict -- all but for LRASPP head
        model.cuda()
        model.load_state_dict(new_dict, strict=False)

    return model
'''






#mnv3 = mobilenetv3.mobilenet_v3_large(pretrained=False, dilated=True)
#lr_mn = lraspp_mobilenet_v3_large_contrast(num_classes=19)
#pdb.set_trace()