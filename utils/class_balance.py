import numpy as np
import pdb 
import time
import torch

def get_class_weights(dataloader, n_classes=19, precomputed=None):

    if precomputed == 'gta_tiny':
        class_weight = None
        return class_weight

    ts = time.time()
    class_freq = np.zeros(n_classes)

    for _, labels in dataloader:
        for c in range(n_classes):
            class_freq[c] += (labels == c).sum()
    
    class_freq /= class_freq.sum()
    class_weight = np.sqrt(np.median(class_freq) / class_freq)
    print('Class weighting time [s]: ' + str(time.time()-ts))
    print(class_weight.round(2))
    return torch.Tensor(class_weight)

# weights rounded to 2 for 100 CS samples (seed 1)
# array([0.16, 0.37, 0.21, 0.87, 1.03, 0.89, 2.24, 1.28, 0.24, 0.93, 0.45, 1.  , 2.79, 0.39, 1.8 , 2.35, 1.97, 5.57, 1.99])

# weights rounded to 2 for 100 CS samples (seed 2)
#array([0.16, 0.38, 0.21, 1.22, 1.  , 0.9 , 2.2 , 1.41, 0.24, 0.85, 0.44, 0.92, 2.6 , 0.38, 1.89, 2.  , 1.56, 2.23, 1.51])

# weights rounded to 2 for 100 CS samples (seed 3)
#array([0.15, 0.33, 0.18, 1.  , 1.15, 0.73, 1.61, 1.17, 0.23, 0.87, 0.43, 0.71, 2.18, 0.36, 2.43, 2.52, 1.51, 3.83, 1.2 ])