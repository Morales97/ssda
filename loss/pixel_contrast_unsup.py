"""
Implementation for the Memory Bank for pixel-level feature vectors and for contrastive loss
from https://github.com/Shathe/SemiSeg-Contrastive 
"""

import torch
import numpy as np
import random
import pdb
import torch.nn.functional as F

class FeatureMemory:
    def __init__(self, num_samples, memory_per_class=256, feature_size=256, n_classes=19):
        self.num_samples = num_samples
        self.memory_per_class = memory_per_class
        self.feature_size = feature_size
        self.memory = [None] * n_classes
        self.n_classes = n_classes
        self.per_class_samples_per_image = max(1, int(round(memory_per_class / num_samples)))
 

    def add_features(self, model, features, class_labels, batch_size):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()

        elements_per_class = batch_size * self.per_class_samples_per_image

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            #selector = model.__getattr__('contrastive_class_selector_' + str(c))  # get the self attention moduel for class c
            features_c = features[mask_c, :] # get features from class c
            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        # get ranking scores
                        '''
                        rank = selector(features_c)
                        rank = torch.sigmoid(rank)
                        # sort them
                        _, indices = torch.sort(rank[:, 0], dim=0)
                        indices = indices.cpu().numpy()
                        features_c = features_c.cpu().numpy()
                        # get features with highest rankings
                        features_c = features_c[indices, :]
                        new_features = features_c[:elements_per_class, :]
                        ''' # NOTE no ranking implemented for the moment -- not using class attention module. Simply select first elements
                        new_features = features_c[:elements_per_class, :].cpu().numpy()
                else:
                    new_features = features_c.cpu().numpy()

                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features

                else: # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.memory_per_class, :]


def contrastive_class_to_class(model, features, class_labels, memory, num_classes=19):
    """
    originally, 'contrastive_class_to_class_learned_memory()' from https://github.com/Shathe/SemiSeg-Contrastive 
    Args:
        model: segmentation model that contains the self-attention MLPs for selecting the features
        to take part in the contrastive learning optimization
        features: Nx256  feature vectors for the contrastive learning (after applying the projection and prediction head)
        class_labels: N corresponding class labels for every feature vector
        num_classes: number of classesin the dataet
        memory: memory bank [List]

    Returns:
        returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
    """

    loss = 0

    for c in range(num_classes):
        # get features of an specific class
        mask_c = class_labels == c
        features_c = features[mask_c,:]
        memory_c = memory[c] # N, 256

        # get the self-attention MLPs both for memory features vectors (projected vectors) and network feature vectors (predicted vectors)
        #selector = model.__getattr__('contrastive_class_selector_' + str(c))
        #selector_memory = model.__getattr__('contrastive_class_selector_memory' + str(c))

        if memory_c is not None and features_c.shape[0] > 1 and memory_c.shape[0] > 1:

            memory_c = torch.from_numpy(memory_c).cuda()

            # L2 normalize vectors
            memory_c = F.normalize(memory_c, dim=1) # N, 256
            features_c_norm = F.normalize(features_c, dim=1) # M, 256

            # compute similarity. All elements with all elements
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))  # MxN
            distances = 1 - similarities # values between [0, 2] where 0 means same vectors
            # M (elements), N (memory)

            '''
            # now weight every sample
            learned_weights_features = selector(features_c.detach()) # detach for trainability
            learned_weights_features_memory = selector_memory(memory_c)

            # self-atention in the memory featuers-axis and on the learning contrsative featuers-axis
            learned_weights_features = torch.sigmoid(learned_weights_features)
            rescaled_weights = (learned_weights_features.shape[0] / learned_weights_features.sum(dim=0)) * learned_weights_features
            rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])
            distances = distances * rescaled_weights

            learned_weights_features_memory = torch.sigmoid(learned_weights_features_memory)
            learned_weights_features_memory = learned_weights_features_memory.permute(1, 0)
            rescaled_weights_memory = (learned_weights_features_memory.shape[0] / learned_weights_features_memory.sum(dim=0)) * learned_weights_features_memory
            rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)
            distances = distances * rescaled_weights_memory
            '''

            loss = loss + distances.mean()

    return loss / num_classes

