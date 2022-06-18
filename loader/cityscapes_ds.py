import os
import sys
import torch
import numpy as np
import pdb 
from PIL import Image
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
from torch.utils import data
import copy 
import shutil

sys.path.append(os.path.abspath('..'))
from loader.loader_utils import pil_loader
from utils.transforms import get_transforms, WeakStrongAug, WeakStrongAug2

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

class cityscapesDataset(data.Dataset):

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
        sample_idxs=None,
        size="tiny",
        unlabeled=False,
        n_augmentations=1,
        do_crop=False,
        hflip=True,
        strong_aug_level = 4,
        downsample_gt = True,
        use_pseudo_labels = False,
        pseudolabel_folder = None
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.split = split
        self.use_pseudo_labels = use_pseudo_labels
        self.pseudolabel_folder = pseudolabel_folder

        self.hflip = hflip
        self.do_crop = True if size == 'small' and split == 'train' else do_crop
        if size == "small":
            self.img_size = (1024, 512) # w, h -- PIL uses (w, h) format
            self.crop_size = (512, 512)
        elif size == "tiny":
            self.img_size = (512, 256)
            self.crop_size = (256, 256)
        else:
            raise Exception('size not valid')
        
        self.orig_size = (2048, 1024)
        self.downsample_gt = downsample_gt

        if not unlabeled:
            self.transforms = get_transforms(aug_level=0)
        if unlabeled:
            weak = get_transforms(aug_level=0)
            strong = get_transforms(aug_level=strong_aug_level)
            if n_augmentations == 1:
                self.transforms = WeakStrongAug(weak, strong)
            if n_augmentations == 2:
                self.transforms = WeakStrongAug2(weak, strong)

        self.n_samples = n_samples
        self.sample_idxs = sample_idxs
        self.n_classes = 19
        self.files = {}
        self.unlabeled = unlabeled

        self.images_base = os.path.join(self.image_path, self.split)
        self.annotations_base = os.path.join(self.label_path, self.split)

        self.files[split] = sorted(recursive_glob(rootdir=self.images_base, suffix=".jpg"))
        if self.sample_idxs is not None:
            files = np.array(self.files[split])
            self.files[split] = files[sample_idxs].tolist()     
        elif self.n_samples >= 0:
            # TODO delete this, only have sample_idxs left
            if not self.unlabeled:
                self.files[split] = self.files[split][:self.n_samples]
                print('Loading %d labeled images and %d unlabeled images' % (len(self.files[split]), 0))
            else:
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
        self.inverted_class_map = dict(zip(range(19), self.valid_classes))
        self.inverted_class_map[self.ignore_index] = self.ignore_index

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        if self.use_pseudo_labels:
            lbl_path = os.path.join(
                self.label_path,
                os.path.basename(img_path)[:-16] + ".png",
            )
        else:
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )
      
        # Load image and segmentation map
        img = pil_loader(img_path, self.img_size[0], self.img_size[1])
        if self.downsample_gt:
            lbl = pil_loader(lbl_path, self.img_size[0], self.img_size[1], is_segmentation=True)
        else:
            lbl = pil_loader(lbl_path, self.orig_size[0], self.orig_size[1], is_segmentation=True)

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

    def generate_pseudolabels(self, model, tau=0.9, ignore_index=250):
        assert self.unlabeled

        model.eval()
        with torch.no_grad():
            for i, img_path in enumerate(self.files[self.split]):
                
                if i % 100 == 0:
                    print(i)

                # get image
                img_path = img_path.rstrip()
                img = pil_loader(img_path, self.img_size[0], self.img_size[1])
                img = self.transforms(img)[0].unsqueeze(0).cuda()

                # generate pseudolabel
                pred = model(img)['out']
                probs = F.softmax(pred, dim=1)
                confidence, pseudo_lbl = torch.max(probs, dim=1)
                pseudo_lbl = torch.where(confidence > tau, pseudo_lbl, ignore_index)

                # save pseudolabel
                lbl_path = os.path.join(
                    './data/cityscapes/pseudo_labels/',
                    self.pseudolabel_folder,
                    os.path.basename(img_path)[:-16] + ".png",
                )
                pseudo_lbl = np.array(pseudo_lbl.cpu(), dtype=np.uint8)
                pseudo_lbl = self.encode_from_trainid_to_id(pseudo_lbl)
                pseudo_lbl_im = Image.fromarray(pseudo_lbl.squeeze(0), mode='L')   
                pseudo_lbl_im.save(lbl_path)

                # How to load the pseudolabel
                lbl = pil_loader(lbl_path, 1024, 512, is_segmentation=True) 
                lbl = np.array(lbl, dtype=np.uint8)
                lbl = self.encode_segmap(lbl)

                # How to viz the pseudolabel
                lbl_col = self.decode_segmap(lbl)
                path2 = lbl_path[:-4] + 'color.png'
                lbl_im = Image.fromarray((lbl_col*255).astype('uint8'), 'RGB')
                lbl_im.save(path2)
                lbl_path_org = os.path.join(
                    self.annotations_base,
                    img_path.split(os.sep)[-2],
                    os.path.basename(img_path)[:-15] + "gtFine_color.png",
                )
                print(path2)
                print(lbl_path_org)
                pdb.set_trace()


    def save_gt_labels(self):
        '''
        save ground truth labels to the pseudolabels folder
        '''
        assert not self.unlabeled 
        for img_path in self.files[self.split]:
            img_path = img_path.rstrip()
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )

            new_lbl_path = os.path.join(
                './data/cityscapes/pseudo_labels/',
                self.pseudolabel_folder,
            )

            # Copy and then rename
            shutil.copy(lbl_path, new_lbl_path)
            original_name = os.path.basename(img_path)[:-15] + "gtFine_labelIds.png"
            new_name = os.path.basename(img_path)[:-16] + ".png"
            shutil.move(os.path.join(new_lbl_path, original_name), os.path.join(new_lbl_path, new_name))
        
        print('Saved %d target labels in pseudolabels folder.' % len(self.files[self.split]))

    def save_pred_viz(self, pred, img_name, img):
        '''
        Save a vizualization of the prediction
        '''
        pred = np.array(pred, dtype=np.uint8)
        pred = self.encode_from_trainid_to_id(pred)
        pred = Image.fromarray(pred.squeeze(0), mode='L')   
        #pred = pred.resize([1024, 512], Image.NEAREST)
        pred = np.array(pred, dtype=np.uint8)
        pred = self.encode_segmap(pred)
        pred = self.decode_segmap(pred)
        pred_im = Image.fromarray((pred*255).astype('uint8'), 'RGB')
        pred_im.save('./data/cityscapes/predictions/' + img_name + '_pred.png')
        img.save('./data/cityscapes/predictions/' + img_name + '_orig.png')

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

    def encode_from_trainid_to_id(self, mask):
        # Put all void classes to zero
        maks_orig = copy.deepcopy(mask)
        for _trainid in range(19):
            mask[maks_orig == _trainid] = self.inverted_class_map[_trainid]
        return mask

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


if __name__ == '__main__':
    '''
    For debugging purposes
    '''
    t_unl_loader = cityscapeDataset(image_path='../data/cityscapes/leftImg8bit_tiny', label_path='../data/cityscapes/gtFine', size="tiny", unlabeled=True, n_samples=0)
    
    from torchvision.utils import save_image
    
    img = t_unl_loader.viz_cr_augment(0)
    save_image(img[0], '/home/danmoral/test_w0.jpg')
    save_image(img[1], '/home/danmoral/test_s0.jpg')
    
    img = t_unl_loader.viz_cr_augment(1)
    save_image(img[0], '/home/danmoral/test_w1.jpg')
    save_image(img[1], '/home/danmoral/test_s1.jpg')
    
    img = t_unl_loader.viz_cr_augment(2)
    save_image(img[0], '/home/danmoral/test_w2.jpg')
    save_image(img[1], '/home/danmoral/test_s2.jpg')
    
    img = t_unl_loader.viz_cr_augment(3)
    save_image(img[0], '/home/danmoral/test_w3.jpg')
    save_image(img[1], '/home/danmoral/test_s3.jpg')
    
    
    '''
    loader = DataLoader(
        s_loader,   
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False, 
    )

    for (images, labels) in loader:
        pdb.set_trace()
    '''

