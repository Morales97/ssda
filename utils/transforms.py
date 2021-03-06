# Borrowed from PAC
# https://github.com/venkatesh-saligrama/PAC 

from __future__ import absolute_import

import random

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
from PIL import Image
from torchvision import transforms
import pdb
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, to_tensor
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
from utils.blur import Blur


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)

def Identity(img, **kwarg):
    return img

def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def color_augment_pool():
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

def color_augment_pool2():
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            #(Rotate, 5, 0),
            (Sharpness, 0.9, 0.05),
            #(ShearX, 0.05, 0),
            #(ShearY, 0.05, 0),
            (Solarize, 256, 0),
            #(TranslateX, 0.05, 0),
            #(TranslateY, 0.05, 0)
            ]
    return augs

def blur_augment_pool():
    augs = ['gaussian',
            'diagonal',
            'diagonal_flip',
            'horizontal',
            'vertical',
            'hpf',
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

class RandAugmentBlur(object):
    def __init__(self, augment_pool, kernel_sizes=[(5,5)]):
        self.augment_pool = augment_pool
        self.kernel_sizes = kernel_sizes
    def __call__(self, img):
        blurs = random.choices(self.augment_pool, k=2)
        for blur_type in blurs:
            kernel_size = random.choices(self.kernel_sizes, k=1)[0]
            blur = Blur(blur_type=blur_type, kernel_size=kernel_size)
            img = blur(img)
        #img = CutoutAbs(img, 16)
        return img


class WeakStrongAug:
    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        img1 = self.base_transform1(x)
        img2 = self.base_transform2(x)
        return [img1, img2]

class WeakStrongAug2:
    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        img1 = self.base_transform1(x)
        img2 = self.base_transform2(x)
        img3 = self.base_transform2(x)
        return [img1, img2, img3]

def get_transforms(crop_size=256, aug_level=0):

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
        # Only crop
        transform_list = [
            transforms.RandomCrop(crop_size),
        ]
    elif aug_level == 3:    # RandAumgent + color jitter
        transform_list = [
            RandAugmentMC(n=2, m=10, augment_pool=color_augment_pool()),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8)
        ]
    elif aug_level == 4:    # RandAumgent + color jitter + blur pool
        transform_list = [
            RandAugmentMC(n=2, m=10, augment_pool=color_augment_pool()),
            transforms.RandomApply([
                RandAugmentBlur(blur_augment_pool()),
            ], p=0.8),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8)
        ]
    elif aug_level == 5:    # RandAumgent + color jitter + gauss blur
        transform_list = [
            RandAugmentMC(n=2, m=10, augment_pool=color_augment_pool()),
            transforms.RandomApply([
                RandAugmentBlur(augment_pool=['gaussian'], kernel_sizes=[(5,5), (7,7), (9,9)]),
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8)
        ]
    elif aug_level == 6:    # RandAumgent2 + color jitter + gauss blur
        transform_list = [
            RandAugmentMC(n=2, m=10, augment_pool=color_augment_pool2()), # more augmentations
            transforms.RandomApply([
                RandAugmentBlur(augment_pool=['gaussian'], kernel_sizes=[(5,5), (7,7), (9,9)]),
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8)
        ]
    elif aug_level == 7:    # only color jitter + gauss blur
        transform_list = [
            transforms.RandomApply([
                RandAugmentBlur(augment_pool=['gaussian'], kernel_sizes=[(5,5), (7,7), (9,9)]),
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8)
        ]
    elif aug_level == 8:    # only RandAugment
        transform_list = [
            RandAugmentMC(n=2, m=10, augment_pool=color_augment_pool()),
        ]
    elif aug_level == 9:
        transform_list = [
            Blur(blur_type='horizontal', kernel_size=(7,7)),
            Blur(blur_type='diagonal', kernel_size=(7,7)),
        ]

    else:
        raise Exception('get_transforms : augmentation not recognized')


    transform_list.append(transforms.ToTensor())
    #transform_list.append(normalize)    
    transform = transforms.Compose(transform_list)

    return transform



def _float_parameter(v, max_v):
    return float(v) * max_v / 10


def _int_parameter(v, max_v):
    return int(v * max_v / 10)




if __name__ == '__main__':
    image = Image.open('/Users/dani/Desktop/sample_img.jpg')
    image = to_tensor(image)

    #save_image(image_blurred, '/Users/dani/Desktop/sample_blurred.jpg')

    pdb.set_trace()
