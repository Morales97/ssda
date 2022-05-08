
'''
from https://github.com/Britefury/cutmix-semisup-seg/blob/master/mask_gen.py
'''

import random

import PIL
import numpy as np
from PIL import Image
from torchvision import transforms
import pdb
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, to_tensor
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

class BoxMaskGenerator ():
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=True, prop_by_area=True, within_bounds=True, invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks


def _cutmix(args, images_s, images_t, labels_s, labels_t):
    assert args.size == 'tiny'
    assert args.batch_size_s == args.batch_size_tl
    
    mask_generator = BoxMaskGenerator((0.25, 0.25))       
    #mask = mask_generator.generate_params(args.batch_size_s, (32,64))  # (B, 1, H, W)
    mask = mask_generator.generate_params(args.batch_size_s, (256,512))  # (B, 1, H, W)
    #up_mask = torch.round(F.interpolate(torch.Tensor(mask), size=(256,512), mode="bilinear", align_corners=False))
    up_mask = torch.Tensor(mask).to('cuda')

    images_s_cutmix = images_s * up_mask + images_t * (1-up_mask)
    images_t_cutmix = images_t * up_mask + images_s * (1-up_mask)   

    up_mask = up_mask.squeeze(1)
    labels_s_cutmix = labels_s * up_mask + labels_t * (1-up_mask)
    labels_t_cutmix = labels_t * up_mask + labels_s * (1-up_mask)    
    return images_s_cutmix, images_t_cutmix, labels_s_cutmix.long(), labels_t_cutmix.long()

def _cutmix_output(args, images, out):
    '''
    Apply CutMix to Images (B, 3, 256, 512) and to upsampled output logits (B, n_classes, 256, 512)
    '''
    assert args.size == 'tiny'
    
    # Generate masks
    mask_generator = BoxMaskGenerator((0.25, 0.25))       
    mask = mask_generator.generate_params(args.batch_size_tu // 2, (256,512))  # (B, 1, H, W)
    mask = torch.Tensor(mask).to('cuda')

    # divide batch in two
    idx_half = images.shape[0] // 2
    images_1 = images[:idx_half]
    images_2 = images[idx_half:]
    out_1 = out[:idx_half]
    out_2 = out[idx_half:]
    assert images_1.shape[0] == images_2.shape[0] 

    # CutMix images
    pdb.set_trace()
    images_1_cutmix = images_1 * mask + images_2 * (1-mask)
    images_2_cutmix = images_2 * mask + images_1 * (1-mask)   
    images_cutmix = torch.cat((images_1_cutmix, images_2_cutmix), dim=0)

    # CutMix output logits
    out_1_cutmix = out_1 * mask + out_2 * (1-mask)
    out_2_cutmix = out_2 * mask + out_1 * (1-mask)    
    out_cutmix = torch.cat((out_1_cutmix, out_2_cutmix), dim=0)

    return images_cutmix, out_cutmix


if __name__ == '__main__':
    image = Image.open('/Users/dani/Desktop/sample_img.jpg')
    #image2 = Image.open('/Users/dani/Desktop/sample_img_gta.jpg')
    image = to_tensor(image)
    #image2 = to_tensor(image2).unsqueeze(0)
    image2 = TF.hflip(image)
    batch = torch.cat((image.unsqueeze(0), image2.unsqueeze(0)), dim=0)
    maskgen = BoxMaskGenerator((0.25, 0.25))       
    mask = maskgen.generate_params(1, (32,64))  # (B, 1, H, W)
    up_mask = torch.round(F.interpolate(torch.Tensor(mask), size=(256,512), mode="bilinear", align_corners=False))

    image_mask = batch * up_mask + torch.cat((batch[1].unsqueeze(0), batch[0].unsqueeze(0)), dim=0) * (1-up_mask)

    save_image(image_mask, '/Users/dani/Desktop/xxx.jpg')

    pdb.set_trace()