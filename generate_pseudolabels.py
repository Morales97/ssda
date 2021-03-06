import os
import random
import shutil
from collections import OrderedDict
import time
import copy
import numpy as np
import torch

from utils.ioutils import parse_args
from model.model import get_model
from loader.cityscapes_ds import cityscapesDataset
import pdb


def main(args):
    torch.set_num_threads(args.max_num_threads)
    print('*** Running generate_pseudolabels.py ...')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print('Seed: ', args.seed)

    # Load trained model
    model = get_model(args)
    model.cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise Exception('No file found at {}'.format(args.resume))
    else:
        raise Exception('No model found to generate pseudolabels with')

    pseudo_lbl_path = _generate_pseudolabels(args, model)


def _generate_pseudolabels(args, model, num_t_samples=2975):
    '''
    Creates a new folder augmented target labeled dataset. Contains:
        - Ground-truth for the few labeled samples
        - Pseudolabels for the unlabeled samples (generated with tau=0.9)
    '''
    n_lbl_samples = args.target_samples
    size = args.size
    assert size in ['tiny', 'small']

    if size == 'tiny':
        image_path_cs = 'data/cityscapes/leftImg8bit_tiny'
    elif size == 'small':
        image_path_cs = 'data/cityscapes/leftImg8bit_small'
    label_path_cs = 'data/cityscapes/gtFine'

    # generate new folder for pseudolabels
    pseudolabel_folder = args.expt_name + '_s' + str(args.seed)
    psuedolabel_path_cs = 'data/cityscapes/pseudo_labels/' + pseudolabel_folder     # NOTE change to your local folder structure
    os.makedirs(psuedolabel_path_cs, exist_ok=True)

    # Select labeled samples 
    if n_lbl_samples != -1:
        idxs = np.arange(num_t_samples)
        idxs = np.random.permutation(idxs)
        idxs_lbl = idxs[:n_lbl_samples]
        idxs_unlbl = idxs[n_lbl_samples:]

    # Save the few labeled samples
    if n_lbl_samples > 0: # if no labeled samples (UDA), skip
        t_lbl_dataset = cityscapesDataset(image_path=image_path_cs, 
                                            label_path=label_path_cs, 
                                            size=size, 
                                            split='train', 
                                            sample_idxs=idxs_lbl)
        # save labels 
        print('Saving labels...')
        t_lbl_dataset.pseudolabel_folder = pseudolabel_folder
        t_lbl_dataset.save_gt_labels()    

    # Generate and save psuedolabels
    t_unlbl_dataset = cityscapesDataset(image_path=image_path_cs, 
                                        label_path=label_path_cs, 
                                        size=size, 
                                        split='train', 
                                        sample_idxs=idxs_unlbl, 
                                        unlabeled=True, 
                                        strong_aug_level=args.aug_level, 
                                        n_augmentations=args.n_augmentations)
    # save pseudolabels
    print('generating pseudolabels...')
    t_unlbl_dataset.pseudolabel_folder = pseudolabel_folder
    t_unlbl_dataset.generate_pseudolabels(model)                 

    return psuedolabel_path_cs


if __name__ == '__main__':
    print('..')
    args = parse_args()
    main(args)

# python generate_pseudolabels.py --seed=3 --expt_name=full_r1 --resume=expts/tmp_last/checkpoint_abl_noPCmix_3.pth.tar