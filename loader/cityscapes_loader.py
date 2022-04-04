import os
import sys
import torch
import numpy as np
import pdb 
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
from loader.loader_utils import pil_loader
import random
from torch.utils import data

sys.path.append(os.path.abspath('..'))
from utils.transforms import get_transforms, WeakStrongAug

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class cityscapesLoader(data.Dataset):

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    label_colours = dict(zip(range(19), colors))

    def __init__(
        self,
        image_path,
        label_path,
        split="train",
        n_samples= -1,        # Select only few samples for training
        size="tiny",
        version="gta",
        test_mode=False,
        rotation=False,
        unlabeled=False,
        do_crop=False,
        hflip=False
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.split = split
        self.rot = rotation
        if size == "small":
            self.img_size = (1024, 512) # w, h -- PIL uses (w, h) format
        elif size == "tiny":
            self.img_size = (512, 256)
            self.crop_size = (256, 256)
        else:
            raise Exception('size not valid')

        if self.rot:
            self.transforms = get_transforms(crop_size=min(self.img_size), split='train', aug_level=1)
            print('Images with random square crops of size ', str(min(self.img_size)))
        else:
            if not unlabeled:
                self.transforms = get_transforms(aug_level=0)
            if unlabeled:
                weak = get_transforms(aug_level=0)
                strong = get_transforms(aug_level=3)
                self.transforms = WeakStrongAug(weak, strong)

        self.n_samples = n_samples
        self.n_classes = 19
        self.files = {}
        self.unlabeled = unlabeled

        self.do_crop = do_crop
        self.hflip = hflip

        self.images_base = os.path.join(self.image_path, self.split)
        self.annotations_base = os.path.join(self.label_path, self.split)

        self.files[split] = sorted(recursive_glob(rootdir=self.images_base, suffix=".jpg"))
        if self.n_samples >= 0:
            if not self.unlabeled:
                self.files[split] = self.files[split][:self.n_samples]
                print('Loading %d labeled images and %d unlabeled images' % (len(self.files[split]), 0))
            else:
                # TODO have labeled and unlabeled in different splits?
                self.files[split] = self.files[split][self.n_samples:]
                

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]


        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        # Rotation pretask
        if self.rot:
            img = pil_loader(img_path, self.img_size[0], self.img_size[1])
            all_rotated_imgs = [
                self.transforms(TF.rotate(img, -90)),
                self.transforms(img),
                self.transforms(TF.rotate(img, 90)),
                self.transforms(TF.rotate(img, 180))]
            all_rotated_imgs = torch.stack(all_rotated_imgs, dim=0)
            rot_lbl = torch.LongTensor([0, 1, 2, 3])
            return all_rotated_imgs, rot_lbl
            
        # Load image and segmentation map
        img = pil_loader(img_path, self.img_size[0], self.img_size[1])
        lbl = pil_loader(lbl_path, self.img_size[0], self.img_size[1], is_segmentation=True)

        # Data Augmentation
        # Crop
        if self.do_crop:
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(img, self.crop_size)
            img = TF.crop(img, i, j, h, w)
            lbl = TF.crop(lbl, i, j, h, w)        

        # Random horizontal flipping
        if self.hflip and random.random() > 0.5:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)

        img = self.transforms(img)
        if self.unlabeled:
            return img

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        classes = np.unique(lbl)
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def viz_cr_augment(self, index):
        img_path = self.files[self.split][index].rstrip()
        img = pil_loader(img_path, self.img_size[0], self.img_size[1])
        
        if False and self.hflip and random.random() > 0.5:
            img = TF.hflip(img)

        img = self.transforms(img)
        if self.unlabeled:
            return img
        else:
            raise Exception('to be used in unlabeled mode')

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

