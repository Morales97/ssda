import numpy as np
import pdb 
import time
import torch

def get_class_weights(dataloader, n_classes=19, precomputed=None, size=None):

    # GTA
    if precomputed == 'gta':
        if size == 'tiny':
            class_weight = torch.Tensor([0.1802,  0.381,   0.2665,  0.7628,  1.2932,       1, 2.9145,  3.6078,  0.3848, 
                                        0.7138,  0.3473,  1.6041,  5.053,   0.5927,  0.9344, 1.6577,  3.8594,  5.357, 12.3351])
        elif size == 'small':
            class_weight = torch.Tensor([0.1808,  0.3848,  0.2801,  0.8211,  1.3222,       1,  2.9463,  3.6512,  0.3896,
                                        0.732,   0.3397,  1.7114,  5.7128,  0.5807,  0.9636,  1.7645,  4.5929,  5.4738, 14.5959])
        print('Retreived GTA class weights for size: ', size)
        return class_weight.to('cuda')

    # Cityscapes
    # Example of cityscapes class weights on 100 labels + 2875 pseudolabels
    #[0.0886 0.3026 0.141  1.0897 1.3093 0.9115 1.7992 1.     0.1587 0.7339
    #0.2549 0.6338 2.1158 0.2336 1.4113 1.2837 1.46   2.6315 1.1661]

    # weights rounded to 2 for 100 CS samples (seed 1)
    # array([0.16, 0.37, 0.21, 0.87, 1.03, 0.89, 2.24, 1.28, 0.24, 0.93, 0.45, 1.  , 2.79, 0.39, 1.8 , 2.35, 1.97, 5.57, 1.99])

    ts = time.time()
    class_freq = np.zeros(n_classes)
    tot_labels = 0
    for _, labels in dataloader:
        for c in range(n_classes):
            class_freq[c] += (labels == c).sum()
        tot_labels += labels.shape[0]
        if tot_labels % 250 == 0:
            print(tot_labels)
        if tot_labels % 5000 == 0:
            break
        
    pdb.set_trace()
    class_freq /= class_freq.sum()
    class_weight = np.sqrt(np.median(class_freq) / class_freq)
    print('Class weighting time [s]: ' + str(time.time()-ts))
    print(class_weight.round(4))
    return torch.Tensor(class_weight).to('cuda')

def get_class_weights_estimation(dataloader_lbl, dataloader_unlbl, model, ema, n_classes=19):
    '''
    estimate target domain class weights by combining labels (L) and predictions (U)
    '''
    ts = time.time()
    class_freq = np.zeros(n_classes)

    # labeled data
    for _, labels in dataloader_lbl:
        for c in range(n_classes):
            class_freq[c] += (labels == c).sum()
        
    # unlabaled data
    class_freq = torch.Tensor(class_freq).to('cuda')
    with ema.average_parameters() and torch.no_grad():
        for images in dataloader_unlbl:
            images = images[0].cuda()
            pred = model(images)['out']
            _, lbl = torch.max(pred, dim=1)
            
            for c in range(n_classes):
                class_freq[c] += (lbl == c).sum()
            

    class_freq /= class_freq.sum()
    class_weight = torch.sqrt(torch.median(class_freq) / class_freq)
    print('Class weighting time [s]: ' + str(time.time()-ts))
    print(class_weight)
    return class_weight


if __name__ == '__main__':
    import os
    print(os.getcwd())
    #path_parent = os.path.dirname(os.getcwd())
    #os.chdir(path_parent)
    print(os.getcwd())
    from loader.cityscapes_ds import cityscapesDataset
    from torch.utils.data import DataLoader

    image_path_cs = 'data/cityscapes/leftImg8bit_small'
    label_path_cs = 'data/cityscapes/gtFine'

    n_lbl_samples = 1
    idxs = np.arange(num_t_samples)
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

# weights rounded to 2 for 100 CS samples (seed 1)
# array([0.16, 0.37, 0.21, 0.87, 1.03, 0.89, 2.24, 1.28, 0.24, 0.93, 0.45, 1.  , 2.79, 0.39, 1.8 , 2.35, 1.97, 5.57, 1.99])

# weights rounded to 2 for 100 CS samples (seed 2)
#array([0.16, 0.38, 0.21, 1.22, 1.  , 0.9 , 2.2 , 1.41, 0.24, 0.85, 0.44, 0.92, 2.6 , 0.38, 1.89, 2.  , 1.56, 2.23, 1.51])

# weights rounded to 2 for 100 CS samples (seed 3)
#array([0.15, 0.33, 0.18, 1.  , 1.15, 0.73, 1.61, 1.17, 0.23, 0.87, 0.43, 0.71, 2.18, 0.36, 2.43, 2.52, 1.51, 3.83, 1.2 ])