import torchvision
import pdb
# from torchsummary import summary


def resnet50_FCN(pretrained=True):
    model = torchvision.models.segmentation.fcn_resnet50(pretrained)

    # change number of classes from 21 to 19
    model.classifier[4] = nn.Conv2d(512, 19, kernel_size=(1,1), stride=(1,1))

    return model