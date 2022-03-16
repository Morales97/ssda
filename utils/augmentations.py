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
        img = CutoutAbs(img, 16)
        return img


def get_augmentations(crop_size=256, split='train', aug_level=0):

    #normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if split == 'test':
        transform_list = [
            transforms.CenterCrop(crop_size),
        ]
    elif split == 'train':
        if aug_level == 0:
            # Mellow transform used for pseudo-labeling during consistency
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size)
            ]
        '''
        elif aug_level == 1:
            # Color Jittering
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
            ]
        elif aug_level == 2:
            # Randaugment
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                RandAugmentMC(n=2, m=10, augment_pool=fixmatch_augment_pool())
            ]
        elif aug_level == 3:
            # Color jittering + Rand augment
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                RandAugmentMC(n=2, m=10, augment_pool=fixmatch_augment_pool()),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8)
            ]
        elif aug_level == 4:
            # lower rotation and sheer augmentations for rotation prediction
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                RandAugmentMC(n=2, m=10, augment_pool=rot_pt_augment_pool()),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8)
            ]
        '''
        else:
            raise Exception('get_transforms : augmentation not recognized')
    else:
        raise Exception('get_transforms: split not recognized')

    if False # data_type == FILELIST:
        # add resize
        transform_list = [transforms.Resize((256, 256))] + transform_list
    else:
        # convert to PIL Image
        transform_list = [transforms.ToPILImage()] + transform_list

    transform_list.append(transforms.ToTensor())
    #transform_list.append(normalize)
    transform = transforms.Compose(transform_list)

    return transform