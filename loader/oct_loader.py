import os
import torch
import numpy as np
import pdb 
from PIL import Image
from loader.loader_utils import pil_loader

from torch.utils import data


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    # DM this will return all images in the folder (e.g., Cirrus_part1) if suffix = .png, 
    # will return all labels if suffix = .npy
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class octLoader(data.Dataset):

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
    ]

    label_colours = dict(zip(range(4), colors))

    def __init__(
        self,
        image_path,
        label_path,
        is_transform=True,
        img_size=(512, 512),    # default is 512x512
        augmentations=None,
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.is_transform = is_transform
        self.split=''
        self.n_classes = 4
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}

        self.images_base = os.path.join(self.image_path, self.split)
        self.annotations_base = os.path.join(self.label_path, self.split)


        self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".png"))
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
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

        if not self.files:
            raise Exception("No files found in %s" % (self.images_base))

        print("Found %d images" % (len(self.files)))

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = pil_loader(img_path, self.img_size[1], self.img_size[0])
        img = np.array(img, dtype=np.uint8)
        #img = img.transpose(2, 0, 1)  # HWC -> CHW

        lbl = pil_loader(lbl_path, self.img_size[1], self.img_size[0], is_segmentation=True)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        # img = img[:, :, ::-1]  # RGB -> BGR. In some conventions BGR is used. Make sure our pre-trained model is RGB
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need to divide by 255.0
            img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        
        classes = np.unique(lbl)
        # lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[1], self.img_size[0]), "nearest", mode="F") # resizing is done by pil_loader
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

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
