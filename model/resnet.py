import torchvision
import pdb
import torch.nn as nn
import torch

from torchsummary import summary

'''
ResNet-50 FCN from torch
'''
def resnet50_FCN(pretrained=False, pretrained_backbone=True):
    model = torchvision.models.segmentation.fcn_resnet50(
        pretrained=pretrained, 
        num_classes=19, 
        pretrained_backbone=pretrained_backbone
    )
    #model.classifier[4] = nn.Conv2d(512, 19, kernel_size=(1,1), stride=(1,1))
    return model

def deeplabv3_rn50(pretrained=False, pretrained_backbone=True):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', 
        pretrained=pretrained, 
        num_classes=19, 
        pretrained_backbone=pretrained_backbone
    )
    #model.classifier[4] = nn.Conv2d(256, 19, kernel_size=(1,1), stride=(1,1))
    return model


def deeplabv3_mobilenetv3_large(pretrained=False, pretrained_backbone=True):
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        pretrained=pretrained,
        num_classes=19,
        pretrained_backbone=pretrained_backbone
    )
    return model

def lraspp_mobilenetv3_large(pretrained=False, pretrained_backbone=True):
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        pretrained=pretrained,
        num_classes=19,
        pretrained_backbone=pretrained_backbone
    )
    return model



'''
modified from
https://github.com/Minauras/road-segmentation/blob/master/ResNet/models.py
'''
def resnet_50_upsampling(pretrained=False, n_classes=19):
    '''
    Has 70M parameters (regular RN50 has 25M)
    
    '''

    res50_conv = nn.Sequential(*list(
            torchvision.models.resnet50(pretrained=pretrained).children())[:-2])  # get all layers except avg-pool & fc


    model = nn.Sequential(
        res50_conv,  # encoder
        nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # 2x upsample NOTE: this has 30M parameters!
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 2x upsample
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 256, kernel_size=6, stride=4, padding=1),  # 4x upsample
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x upsample
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0),  # logits per pixel
        nn.Sigmoid()  # predictions per pixel  # could remove and use BCEWithLogitsLoss instead of BCELoss.
    )

    # init
    for p in model.parameters():
        try:
            nn.init.xavier_normal_(p)
        except ValueError:
            pass

    return model

def resnet_34_upsampling(pretrained=True, n_classes=19):
    '''
    
    '''
    res50_conv = nn.Sequential(*list(
            torchvision.models.resnet34(pretrained=pretrained).children())[:-2])  # get all layers except avg-pool & fc

    model = nn.Sequential(
        res50_conv,  # encoder
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 2x upsample
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x upsample
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 128, kernel_size=6, stride=4, padding=1),  # 4x upsample
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 2x upsample
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0),  # logits per pixel
        nn.Sigmoid()  # predictions per pixel  # could remove and use BCEWithLogitsLoss instead of BCELoss.
    )

    # init
    for p in model.parameters():
        try:
            nn.init.xavier_normal_(p)
        except ValueError:
            pass

    return model


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05, hidden=[]):
        super(Predictor, self).__init__()
        layer_nodes = [inc] + hidden + [num_class]
        layers = [nn.Linear(layer_nodes[i], layer_nodes[i+1]) for i in range(len(layer_nodenums)-1)]

        self.fc = nn.Sequential(*layers)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x_out = self.fc(x) / self.temp
        return x_out

class RotationPred(nn.Module):
    def __init__(self, backbone, clas_head):
        super().__init__()
        self.backbone = backbone
        self.classifier = clas_head

    def forward(self, x):
        x = self.backbone(x)
        x = x.squeeze()     # (B·4, inc, 1, 1) -> (B·4, inc)
        x = self.classifier(x)
        return x

'''
mnv3 = torchvision.models.mobilenet_v3_large(pretrained=False)
mnv3_d = torchvision.models.mobilenet_v3_large(pretrained=False, dilated=True)
lraspp_mn = lraspp_mobilenetv3_large()
dl_mobilenet = deeplabv3_mobilenetv3_large()
deeplab = deeplabv3_rn50()
rn50_fcn = resnet50_FCN()
rn50_u = resnet_50_upsampling()
rn34_u = resnet_34_upsampling()
pdb.set_trace()
'''