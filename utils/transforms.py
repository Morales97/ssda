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
            # Mellow transform used for pseudo-labeling during consistency
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
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