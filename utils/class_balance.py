import numpy as np
import pdb 

def get_class_weights(dataloader, n_classes=19):

    class_freq = np.zeros(n_classes)

    for _, labels in dataloader:
        for c in range(n_classes):
            class_freq[c] += (labels == c).sum()
    
    class_freq /= class_freq.sum()
    class_weight = np.sqrt(np.median(class_freq) / class_freq)

    pdb.set_trace()

# weights rounded to 2 for 100 CS samples (seed 1)
# array([0.16, 0.37, 0.21, 0.87, 1.03, 0.89, 2.24, 1.28, 0.24, 0.93, 0.45, 1.  , 2.79, 0.39, 1.8 , 2.35, 1.97, 5.57, 1.99])