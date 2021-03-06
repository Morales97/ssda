
import os
import random
import shutil
from collections import OrderedDict
import time
import copy
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import get_model
from loader.loaders import get_loaders, get_loaders_pseudolabels, get_source_test_loader
from utils.ioutils import parse_args
import pdb

def main(args):
    torch.set_num_threads(args.max_num_threads)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print('Seed: ', args.seed)
    

    args.batch_size_tl = 1
    '''
    # Load data
    if args.pseudolabel_folder is None:
        source_loader, target_loader, target_loader_unl, val_loader, _ = get_loaders(args)
    else:
        source_loader, target_loader, target_loader_unl, val_loader, _ = get_loaders_pseudolabels(args)
    '''
    source_test_loader = get_source_test_loader(args)

    # Init model and EMA
    model = get_model(args)
    model.cuda()
    model.train()
    '''
    ema_model = get_model(args)
    ema_model.cuda()
    ema_model.train()
    for param in ema_model.parameters():
        param.detach_()
    '''

    # To resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            #ema_model.load_state_dict(checkpoint['ema_state_dict'])
            print('*** Loading checkpoint from ', args.resume)

        else:
            raise Exception('No file found at {}'.format(args.resume))

    model.eval()

    with torch.no_grad():
        n_class = 19
        num_samples = 100
        dim_emb = 2048

        X = np.zeros((n_class, num_samples, dim_emb))
        Y = np.zeros((n_class, num_samples))
        class_ptr = np.zeros((n_class)).astype(int)
        class_count = 0

        #for image, label in val_loader:
        for image, label in source_test_loader:
            image = image.cuda()
            label = label.cuda()
            input_shape = label.shape[1:]

            out = model(image)
            feat = out['feat']
            feat = F.interpolate(feat, size=input_shape, mode="bilinear", align_corners=False)

            feat = feat.cpu().numpy()
            label = label.cpu().numpy()
            img_classes = np.unique(label)

            # select one pixel from every class present in the image. repeat until all pixel classes are filled
            for c in range(19):
                if class_ptr[c] == num_samples or (c not in img_classes):
                    pass
                else:
                    idxs = np.where(label == c)

                    n_samples = min(10, num_samples - class_ptr[c])
                    sample_ids = np.random.choice(len(idxs[0]), n_samples) 
                    for sample_id in sample_ids:
                        X[c, class_ptr[c]] = feat[0, :, int(idxs[1][sample_id]), int(idxs[2][sample_id])]
                        Y[c, class_ptr[c]] = c     # we dont need Y at all
                        class_ptr[c] += 1
                    
                    if class_ptr[c] == num_samples:
                        class_count += 1
                        print('Class count: ', class_count)
                        if class_count == n_class:
                            X = X.reshape(-1, dim_emb)
                            Y = Y.reshape(-1)
                            #np.savetxt('X_cityscapes.txt', X)
                            #np.savetxt('Y_cityscapes.txt', Y)
                            np.savetxt('X_gta.txt', X)
                            np.savetxt('Y_gta.txt', Y)
                            print('txt files saved')
                            return


if __name__ == '__main__':
    print('Running tsne.py')
    args = parse_args()
    main(args)

# python tsne.py --resume=expts/tmp_last/checkpoint_abl_noPCmix_p2_3.pth.tar