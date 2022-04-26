
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

def lab_transform_batch(source_batch, target_image):
    """
    Transform a batch of images into the LAB color space of a target image
    """
    assert target_image.shape[0] == 3 and source_batch.shape[1] == 3
    target_image = rgb_to_lab(target_image)
    source_batch = rgb_to_lab(source_batch)

    mean_t = target_image.mean(axis=[1,2]).view(1, -1, 1, 1)
    std_t = target_image.std(axis=[1,2]).view(1, -1, 1, 1)
    mean_s = source_batch.mean(axis=[0,2,3]).view(1, -1, 1, 1)
    std_s = source_batch.std(axis=[0,2,3]).view(1, -1, 1, 1)

    source_batch = (source_batch - mean_s) / std_s * std_t + mean_t
    source_batch = lab_to_rgb(source_batch)
    return source_batch

def lab_transform(source_batch, target_batch):
    """
    Transform a batch of images into the LAB color space of target images, each image differently
    """
    assert target_batch.shape[1] == 3 and source_batch.shape[1] == 3
    target_batch = rgb_to_lab(target_batch)
    source_batch = rgb_to_lab(source_batch)

    for i in range(target_batch.shape[0]):
        mean_t = target_batch[i].mean(axis=[1,2]).view(-1, 1, 1)
        std_t = target_batch[i].std(axis=[1,2]).view(-1, 1, 1)
        mean_s = source_batch[i].mean(axis=[1,2]).view(-1, 1, 1)
        std_s = source_batch[i].std(axis=[1,2]).view(-1, 1, 1)

        source_batch[i] = (source_batch[i] - mean_s) / (std_s + 1e-4) * std_t + mean_t
    
    source_batch = lab_to_rgb(source_batch)
    return source_batch

if __name__ == '__main__':
    test_on_local()




    