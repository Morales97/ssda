import numpy as np
from torch.utils.data import DataLoader
from loader.cityscapes_ds import cityscapesDataset
from loader.gta_ds import gtaDataset
import pdb
import torch
import torch.nn.functional as F
import os

def get_loaders(args, num_t_samples=2975):
    
    n_lbl_samples = args.target_samples
    size = args.size
    assert size in ['tiny', 'small']

    if size == 'tiny':
        image_path_gta = 'data/gta5/images_tiny'
        #image_path_gta = 'data/gta5/gta5cycada/images_tiny'
        image_path_cs = 'data/cityscapes/leftImg8bit_tiny'
        #print('*** GTA5 stylized as CS with CyCada ***')
    elif size == 'small':
        image_path_gta = 'data/gta5/images_small'
        #image_path_gta = 'data/gta5/gta5cycada/images_small'
        image_path_cs = 'data/cityscapes/leftImg8bit_small'
        #print('*** GTA5 stylized as CS with CyCada ***')
    label_path_gta = 'data/gta5/labels'
    label_path_cs = 'data/cityscapes/gtFine'

    # Select randomly labelled samples 
    if n_lbl_samples != -1:
        idxs = np.arange(num_t_samples)
        idxs = np.random.permutation(idxs)
        idxs_lbl = idxs[:n_lbl_samples]
        idxs_unlbl = idxs[n_lbl_samples:]

    # Get Source loader
    s_dataset = gtaDataset(image_path=image_path_gta, label_path=label_path_gta, size=size, split="all_gta")
    s_loader = DataLoader(
        s_dataset,
        batch_size=args.batch_size_s,
        num_workers=args.num_workers,
        shuffle=True,
    )
    print('Loading %d source domain images, labelled, from %s' % (len(s_dataset), image_path_gta))

    # Get Target loader(s)
    # Fully supervised
    if n_lbl_samples == -1:     
        t_dataset = cityscapesDataset(image_path=image_path_cs, label_path=label_path_cs, size=size, split='train')
        t_lbl_loader = DataLoader(
            t_dataset,
            batch_size=args.batch_size_tl,
            num_workers=args.num_workers,
            shuffle=True,
        ) 
        print('Loading %d target domain images, labelled, from %s' % (len(t_dataset), image_path_cs))
        print('No target domain unlabelled images')
        t_unlbl_loader = None

    # Semi-supervised
    else:                       
        t_lbl_dataset = cityscapesDataset(image_path=image_path_cs, 
                                          label_path=label_path_cs, 
                                          size=size, 
                                          split='train', 
                                          sample_idxs=idxs_lbl)
                                          
        t_unlbl_dataset = cityscapesDataset(image_path=image_path_cs, 
                                            label_path=label_path_cs, 
                                            size=size, 
                                            split='train', 
                                            sample_idxs=idxs_unlbl, 
                                            unlabeled=True, 
                                            strong_aug_level=args.aug_level, 
                                            n_augmentations=args.n_augmentations)
        t_lbl_loader = DataLoader(
            t_lbl_dataset,
            batch_size=args.batch_size_tl,
            num_workers=args.num_workers,
            shuffle=True,
        ) 
        t_unlbl_loader = DataLoader(
            t_unlbl_dataset,
            batch_size=args.batch_size_tu,
            num_workers=args.num_workers,
            shuffle=True,
        ) 
        print('Loading %d target domain images, labelled, from %s' % (len(t_lbl_dataset), image_path_cs))
        print('Loading %d target domain images, unlabelled, from %s' % (len(t_unlbl_dataset), image_path_cs))


    # Get validation loader
    val_dataset = cityscapesDataset(image_path=image_path_cs, label_path=label_path_cs, size=size, split='val', hflip=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_tl,
        num_workers=args.num_workers,
        shuffle=False,
    )    
    

    return s_loader, t_lbl_loader, t_unlbl_loader, val_loader #, idxs, idxs_lbl, idxs_unlbl
    


def get_loaders_pseudolabels(args, num_t_samples=2975):
    '''
    Returns
        - Source labeled loader
        - Target labeled loader (100 labels + 2875 pseudolabels)
        - Target unlabeled loader (2875 images without labels)
    '''
    
    n_lbl_samples = args.target_samples
    
    size = args.size
    assert size in ['tiny', 'small']

    if size == 'tiny':
        image_path_gta = 'data/gta5/images_tiny'
        #image_path_gta = 'data/gta5/gta5cycada/images_tiny'
        image_path_cs = 'data/cityscapes/leftImg8bit_tiny'
        #print('*** GTA5 stylized as CS with CyCada ***')
    elif size == 'small':
        image_path_gta = 'data/gta5/images_small'
        #image_path_gta = 'data/gta5/gta5cycada/images_small'
        image_path_cs = 'data/cityscapes/leftImg8bit_small'
        #print('*** GTA5 stylized as CS with CyCada ***')
    label_path_gta = 'data/gta5/labels'
    label_path_cs = 'data/cityscapes/gtFine'
    pseudolabel_path_cs = os.path.join('data/cityscapes/pseudo_labels/', args.pseudolabel_folder)

    # Select randomly labelled samples 
    if n_lbl_samples != -1:
        idxs = np.arange(num_t_samples)
        idxs = np.random.permutation(idxs)
        idxs_lbl = idxs[:n_lbl_samples]
        idxs_unlbl = idxs[n_lbl_samples:]

    # *** Source loader
    s_dataset = gtaDataset(image_path=image_path_gta, label_path=label_path_gta, size=size, split="all_gta")
    s_loader = DataLoader(
        s_dataset,
        batch_size=args.batch_size_s,
        num_workers=args.num_workers,
        shuffle=True,
    )
    print('Loading %d source domain images, labelled, from %s' % (len(s_dataset), image_path_gta))

    # *** Target loader(s)
    # pseudo_dataset has 100 labels + 2875 pseudolabels
    t_pseudo_dataset = cityscapesDataset(image_path=image_path_cs, 
                                        label_path=pseudolabel_path_cs, 
                                        size=size, 
                                        split='train', 
                                        use_pseudo_labels=True)      

    t_unlbl_dataset = cityscapesDataset(image_path=image_path_cs, 
                                        label_path=label_path_cs, 
                                        size=size, 
                                        split='train', 
                                        sample_idxs=idxs_unlbl, 
                                        unlabeled=True, 
                                        strong_aug_level=args.aug_level, 
                                        n_augmentations=args.n_augmentations)              
    
    t_pseudo_loader = DataLoader(
        t_pseudo_dataset,
        batch_size=args.batch_size_tl,
        num_workers=args.num_workers,
        shuffle=True,
    ) 
    t_unlbl_loader = DataLoader(
        t_unlbl_dataset,
        batch_size=args.batch_size_tu,
        num_workers=args.num_workers,
        shuffle=True,
    ) 
    print('Loading %d target domain images, labels + pseudolabels, from %s' % (len(t_pseudo_dataset), image_path_cs))
    print('Loading %d target domain images, unlabelled, from %s' % (len(t_unlbl_dataset), image_path_cs))

    # Get validation loader
    val_dataset = cityscapesDataset(image_path=image_path_cs, label_path=label_path_cs, size=size, split='val', hflip=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_tl,
        num_workers=args.num_workers,
        shuffle=False,
    )    
    
    return s_loader, t_pseudo_loader, t_unlbl_loader, val_loader #, idxs, idxs_lbl, idxs_unlbl



def generate_pseudolabels(args, model, ema, num_t_samples=2975):
    n_lbl_samples = args.target_samples
    size = args.size
    assert size in ['tiny', 'small']

    if size == 'tiny':
        image_path_cs = 'data/cityscapes/leftImg8bit_tiny'
    elif size == 'small':
        image_path_cs = 'data/cityscapes/leftImg8bit_small'
    label_path_gta = 'data/gta5/labels'
    label_path_cs = 'data/cityscapes/gtFine'

    # Select randomly labelled samples 
    if n_lbl_samples != -1:
        idxs = np.arange(num_t_samples)
        idxs = np.random.permutation(idxs)
        idxs_lbl = idxs[:n_lbl_samples]
        idxs_unlbl = idxs[n_lbl_samples:]

    # *** Target loader(s)
    t_lbl_dataset = cityscapesDataset(image_path=image_path_cs, 
                                        label_path=label_path_cs, 
                                        size=size, 
                                        split='train', 
                                        sample_idxs=idxs_lbl)
                                        
    t_unlbl_dataset = cityscapesDataset(image_path=image_path_cs, 
                                        label_path=label_path_cs, 
                                        size=size, 
                                        split='train', 
                                        sample_idxs=idxs_unlbl, 
                                        unlabeled=True, 
                                        strong_aug_level=args.aug_level, 
                                        n_augmentations=args.n_augmentations)
    
    # generate new folder for pseudolabels
    pseudolabel_folder = args.expt_name + str(args.seed) + '_test'
    psuedolabel_path_cs = 'data/cityscapes/pseudo_labels/' + pseudolabel_folder
    t_lbl_dataset.pseudolabel_folder = pseudolabel_folder
    t_unlbl_dataset.pseudolabel_folder = pseudolabel_folder
    os.makedirs(psuedolabel_path_cs, exist_ok=True)

    # save labels 
    print('Saving labels...')
    t_lbl_dataset.save_gt_labels()

    # save pseudolabels
    print('generating pseudolabels...')
    t_unlbl_dataset.generate_pseudolabels(model, ema)                 

    return psuedolabel_path_cs

    
     