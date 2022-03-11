import torchvision
import pdb
# from torchsummary import summary


def resnet50_FCN(pretrained=True):
    model = torchvision.models.segmentation.fcn_resnet50(pretrained)
    return model