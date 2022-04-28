import numpy as np
from torch.utils.data import DataLoader
from loader.cityscapes_ds import cityscapesDataset
from loader.gta_ds import gtaDataset
import pdb


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
        #image_path_gta = 'data/gta5/images_small'
        image_path_gta = 'data/gta5/gta5cycada/images_small'
        image_path_cs = 'data/cityscapes/leftImg8bit_small'
        print('*** GTA5 stylized as CS with CyCada ***')
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
    