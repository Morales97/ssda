
from __future__ import absolute_import

import torch
#from loader.cityscapes_ds import cityscapesDataset
#from loader.gta_ds import gtaDataset
import numpy as np
from PIL import Image
from kornia.color import rgb_to_lab, lab_to_rgb
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, to_tensor
import pdb
from torchvision.utils import save_image



def test_on_cluster():
    '''
    TODO this function is halfway done
    '''

    size = 'tiny'
    if size == 'tiny':
        image_path_gta = 'data/gta5/images_tiny'
        image_path_cs = 'data/cityscapes/leftImg8bit_tiny'
    elif size == 'small':
        image_path_gta = 'data/gta5/images_small'
        image_path_cs = 'data/cityscapes/leftImg8bit_small'
    label_path_gta = 'data/gta5/labels'
    label_path_cs = 'data/cityscapes/gtFine'

    # Get Source loader -- val set (2500 images)
    s_dataset = gtaDataset(image_path=image_path_gta, label_path=label_path_gta, size=size, split="val")
    s_loader = DataLoader(
        s_dataset,
        batch_size=args.batch_size_s,
        num_workers=args.num_workers,
        shuffle=True,
    )

    # Get target loader -- 100 samples
    t_dataset = cityscapesDataset(image_path=image_path_cs, label_path=label_path_cs, size=size, split='train', 
                                    sample_idxs=np.arange(100))
    t_lbl_loader = DataLoader(
        t_dataset,
        batch_size=args.batch_size_tl,
        num_workers=args.num_workers,
        shuffle=True,
    ) 

def test_on_local():
    image_cs = Image.open('/Users/dani/Desktop/sample_img.jpg')
    image_cs = to_tensor(image_cs)

    image_cs = rgb_to_lab(image_cs)

    mean_cs = image_cs.mean(axis=[1,2]).view(-1, 1, 1)
    std_cs = image_cs.std(axis=[1,2]).view(-1, 1, 1)

    image_gta = Image.open('/Users/dani/Desktop/sample_img_gta.jpg')
    image_gta = to_tensor(image_gta)
    image_gta = rgb_to_lab(image_gta)
    mean_gta = image_gta.mean(axis=[1,2]).view(-1, 1, 1)
    std_gta = image_gta.std(axis=[1,2]).view(-1, 1, 1)

    image_gta = image_gta - mean_gta
    image_gta = image_gta / std_gta
    image_gta = image_gta * std_cs
    image_gta = image_gta + mean_cs

    image_gta = lab_to_rgb(image_gta)

    save_image(image_gta, '/Users/dani/Desktop/image_gta_lab.jpg')

if __name__ == '__main__':
    test_on_local()




    