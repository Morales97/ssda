import numpy as np
import pdb 

def get_class_weights(dataloader, n_classes=19):

    class_freq = np.zeros(n_classes)

    for _, labels in dataloader:
        for c in range(n_classes):
            class_freq[c] += (labels == c).sum()
    
    class_freq /= class_freq.sum()
    pdb.set_trace()
