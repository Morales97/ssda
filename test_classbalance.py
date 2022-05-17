import numpy as np
import pdb 
import time
import torch
from utils.class_balance import get_class_weights


if __name__ == '__main__':

    from loader.cityscapes_ds import cityscapesDataset
    from torch.utils.data import DataLoader

    image_path_cs = 'data/cityscapes/leftImg8bit_small'
    label_path_cs = 'data/cityscapes/gtFine'

    n_lbl_samples = 1
    idxs = np.arange(2975)
    #idxs = np.random.permutation(idxs) # do not shuffle
    idxs_lbl = idxs[:n_lbl_samples]

    t_lbl_dataset = cityscapesDataset(image_path=image_path_cs, 
                                        label_path=label_path_cs, 
                                        size='small', 
                                        split='train', 
                                        sample_idxs=idxs_lbl)
    t_lbl_loader = DataLoader(
        t_lbl_dataset,
        batch_size=1,
        num_workers=1)

    get_class_weights(t_lbl_loader)