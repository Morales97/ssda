import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from loader.cityscapes_ds import cityscapesDataset
from loader.gta_ds import gtaDataset
import pdb

def _build_size(orig_img, width, height):
    size = [width, height]
    if size[0] == -1: size[0] = orig_img.width
    if size[1] == -1: size[1] = orig_img.height
    return size


# 3Gb / 300k = 10000 (per worker)
# @lru_cache(maxsize=5000)
def _load_lru_cache(*args, **kwargs):
    return _load(*args, **kwargs)


def _load(_path, is_segmentation, resize, width, height, convert_segmentation=True):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(_path, 'rb') as f:
        with Image.open(f) as _img:
            if is_segmentation:
                if convert_segmentation:
                    #_img = _img.convert()
                    pass    # for GTA-5 dataset, it should not convert anything!. The default label file comes with the trianing IDs!
                if resize: _img = _img.resize(_build_size(_img, width, height), Image.NEAREST)
            else:
                _img = _img.convert('RGB')
                if resize: _img = _img.resize(_build_size(_img, width, height), Image.ANTIALIAS)
    # print(np.asarray(_img).nbytes/1e6)
    return _img


def pil_loader(path, std_width, std_height, is_segmentation=False, lru_cache=False, convert_segmentation=True):
    if lru_cache:
        load_fn = _load_lru_cache
    else:
        load_fn = _load
    return load_fn(path, is_segmentation, True, std_width, std_height, convert_segmentation=convert_segmentation)


def get_loaders(args, num_t_samples=2975, size='tiny'):
    image_path_gta = 'data/gta5/images_tiny'
    label_path_gta = 'data/gta5/labels'
    image_path_cs = 'data/cityscapes/leftImg8bit_tiny'
    label_path_cs = 'data/cityscapes/gtFine'
    n_lbl_samples = args.target_samples
 
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
        t_unlbl_loader = None
    # Semi-supervised
    else:                       
        t_lbl_dataset = cityscapesDataset(image_path=image_path_cs, label_path=label_path_cs, size=size, split='train', sample_idxs=idxs_lbl)
        t_unlbl_dataset = cityscapesDataset(image_path=image_path_cs, label_path=label_path_cs, size=size, split='train', sample_idxs=idxs_unlbl, unlabeled=True)
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

    # Get validation loader
    val_dataset = cityscapesDataset(image_path=image_path_cs, label_path=label_path_cs, size=size, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_tl,
        num_workers=args.num_workers,
        shuffle=True,
    )    
    

    return s_loader, t_lbl_loader, t_unlbl_loader, val_loader, idxs, idxs_lbl, idxs_unlbl