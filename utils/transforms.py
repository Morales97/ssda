# Borrowed from PAC
# https://github.com/venkatesh-saligrama/PAC 


import random

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
from PIL import Image
from torchvision import transforms


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)



def cr_augment_pool():
    augs = [# (AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            #(Equalize, None, None),
            #(Identity, None, None),
            #(Posterize, 4, 4),
            #(Rotate, 5, 0),
            (Sharpness, 0.9, 0.05),
            #(ShearX, 0.05, 0),
            #(ShearY, 0.05, 0),
            (Solarize, 256, 0),
            #(TranslateX, 0.05, 0),
            #(TranslateY, 0.05, 0)
            ]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m, augment_pool):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = augment_pool

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        #img = CutoutAbs(img, 16)
        return img


class WeakStrongAug:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        img1 = self.base_transform1(x)
        img2 = self.base_transform2(x)
        return [img1, img2]


def get_transforms(crop_size=256, split='train', aug_level=0):

    #normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if split == 'test':
        transform_list = [
            transforms.CenterCrop(crop_size),
        ]
    elif split == 'train':
        if aug_level == 0:
            # Do nothing
            transform_list = []
        elif aug_level == 1:
            # Mellow transform used for rotation pretrain
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
            ]
        elif aug_level == 2:
            # Strong augmentation for CR: Color Jitter + RandAugment
            transform_list = [
                RandAugmentMC(n=2, m=10, augment_pool=cr_augment_pool()),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8)
            ]

        # NOTE see https://github.com/venkatesh-saligrama/PAC for more possible augmentations
        else:
            raise Exception('get_transforms : augmentation not recognized')
    else:
        raise Exception('get_transforms: split not recognized')


    transform_list.append(transforms.ToTensor())
    #transform_list.append(normalize)
    transform = transforms.Compose(transform_list)

    return transform


# TODO what is this? wouldn't it be better to have it random?
def _float_parameter(v, max_v):
    return float(v) * max_v / 10