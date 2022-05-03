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

            use_selector=True
            if use_selector:
                selector = model.__getattr__('contrastive_class_selector_' + str(c))  # get the self attention moduel for class c
                features_c = features[mask_c, :] # get features from class c
                if features_c.shape[0] > 0:
                    if features_c.shape[0] > elements_per_class:
                        with torch.no_grad():
                            # get ranking scores
                            rank = selector(features_c)
                            rank = torch.sigmoid(rank)
                            # sort them
                            _, indices = torch.sort(rank[:, 0], dim=0)
                            indices = indices.cpu().numpy()
                            features_c = features_c.cpu().numpy()
                            # get features with highest rankings
                            features_c = features_c[indices, :]
                            new_features = features_c[:elements_per_class, :]
                    else:
                        new_features = features_c.cpu().numpy()

                    if self.memory[c] is None: # was empy, first elements
                        self.memory[c] = new_features

                    else: # add elements to already existing list
                        # keep only most recent memory_per_class samples
                        self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.memory_per_class, :]

            else:
                features_c = features[mask_c, :] # get features from class c
                if features_c.shape[0] > 0:
                    if features_c.shape[0] > elements_per_class:
                        with torch.no_grad():
                            # NOTE no ranking implemented for the moment -- not using class attention module. Simply select first elements
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

            use_prototypes = False
            if use_prototypes:
                prototype = features_c.mean(dim=0).unsqueeze(0)  # M=1, 256
                features_c_norm = F.normalize(prototype, dim=1) # M=1, 256
            else:
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


class AlonsoContrastiveLearner:
    def __init__(self, num_samples):
        self.feature_memory = FeatureMemory(num_samples)


    def add_features_to_memory(self, outputs_t_ema, labels_t, model):
        prob_t, pred_t = torch.max(torch.softmax(outputs_t_ema['out'], dim=1), dim=1)  

        # save the projected features if the prediction is correct and more confident than 0.95
        # the projected features are not upsampled, it is a lower resolution feature map. Downsample labels and preds (x8)
        proj_t = outputs_t_ema['proj']
        labels_t_down = F.interpolate(labels_t.unsqueeze(0).float(), size=(proj_t.shape[2], proj_t.shape[3]), mode='nearest').squeeze()
        pred_t_down = F.interpolate(pred_t.unsqueeze(0).float(), size=(proj_t.shape[2], proj_t.shape[3]), mode='nearest').squeeze()
        prob_t_down = F.interpolate(prob_t.unsqueeze(0), size=(proj_t.shape[2], proj_t.shape[3]), mode='nearest').squeeze()
        
        mask = ((pred_t_down == labels_t_down).float() * (prob_t_down > 0.95).float()).bool() # (B, 32, 64)
        labels_t_down_selected = labels_t_down[mask]

        proj_t = proj_t.permute(0,2,3,1)    # (B, 32, 64, C)
        proj_t_selected = proj_t[mask, :]
        print(proj_t_selected.shape[0])
        if proj_t_selected.shape[0] > 0:
            self.feature_memory.add_features(model, proj_t_selected, labels_t_down_selected, args.batch_size_tl)

        store_S_pixels = False  # Results are better when only storing features from T, not S+T. This is also what Alonso et al does
        if store_S_pixels:
            with ema.average_parameters() and torch.no_grad():  
                outputs_s = model(images_s) 

            prob_s, pred_s = torch.max(torch.softmax(outputs_s['out'], dim=1), dim=1)  
            proj_s = outputs_s['proj']
            labels_s_down = F.interpolate(labels_s.unsqueeze(0).float(), size=(proj_s.shape[2], proj_s.shape[3]), mode='nearest').squeeze()
            pred_s_down = F.interpolate(pred_s.unsqueeze(0).float(), size=(proj_s.shape[2], proj_s.shape[3]), mode='nearest').squeeze() 
            prob_s_down = F.interpolate(prob_s.unsqueeze(0), size=(proj_s.shape[2], proj_s.shape[3]), mode='nearest').squeeze()

            mask = ((pred_s_down == labels_s_down).float() * (prob_s_down > 0.95).float()).bool() # (B, 32, 64)
            labels_s_down_selected = labels_s_down[mask]

            proj_s = proj_s.permute(0,2,3,1)    # (B, 32, 64, C)
            proj_s_selected = proj_s[mask, :]
            
            if proj_s_selected.shape[0] > 0:
                self.feature_memory.add_features(model, proj_s_selected, labels_s_down_selected, args.batch_size_s)


    def labeled_pc(self, outputs_s, outputs_t, labels_s, labels_t):
        loss_labeled = 0
        pred_s = outputs_s['pred']
        pred_tl = outputs_t['pred']

        use_s = False
        if use_s:

            labels_s_down = F.interpolate(labels_s.unsqueeze(0).float(), size=(pred_s.shape[2], pred_s.shape[3]), mode='nearest').squeeze()
            ignore_label = 250
            mask = (labels_s_down != ignore_label)
            
            use_threhsold_s = True
            if use_threhsold_s:
                prob, pseudo_lbl = torch.max(F.softmax(outputs_s['out'], dim=1).detach(), dim=1)
                pseudo_lbl_down = F.interpolate(pseudo_lbl.unsqueeze(0).float(), size=(pred_s.shape[2], pred_s.shape[3]), mode='nearest').squeeze()
                prob_down = F.interpolate(prob.unsqueeze(0), size=(pred_s.shape[2], pred_s.shape[3]), mode='nearest').squeeze()     
                threshold = 0.9
                mask = prob_down > threshold      
                mask = mask * (labels_s_down == pseudo_lbl_down)     
            
            pred_s = pred_s.permute(0, 2, 3, 1)
            pred_s = pred_s[mask, ...]
            labels_s_down = labels_s_down[mask]

            loss_labeled = loss_labeled + contrastive_class_to_class(None, pred_s, labels_s_down, self.feature_memory.memory)

        use_tl = False
        if use_tl:

            labels_t_down = F.interpolate(labels_t.unsqueeze(0).float(), size=(pred_tl.shape[2], pred_tl.shape[3]), mode='nearest').squeeze()
            ignore_label = 250
            mask = (labels_t_down != ignore_label)
            
            use_threhsold_tl = True
            if use_threhsold_tl:
                prob, pseudo_lbl = torch.max(F.softmax(outputs_t['out'], dim=1).detach(), dim=1)
                pseudo_lbl_down = F.interpolate(pseudo_lbl.unsqueeze(0).float(), size=(pred_tl.shape[2], pred_tl.shape[3]), mode='nearest').squeeze()
                prob_down = F.interpolate(prob.unsqueeze(0), size=(pred_tl.shape[2], pred_tl.shape[3]), mode='nearest').squeeze()     
                threshold = 0.9
                mask = prob_down > threshold      
                mask = mask * (labels_t_down == pseudo_lbl_down)     
            
            pred_tl = pred_tl.permute(0, 2, 3, 1)
            pred_tl = pred_tl[mask, ...]
            labels_t_down = labels_t_down[mask]

            loss_labeled = loss_labeled + contrastive_class_to_class(None, pred_tl, labels_t_down, self.feature_memory.memory)

        return loss_labeled

    def unlabeled_pc(self, outputs_tu):
        pred_tu = outputs_tu['pred']

        # compute pseudolabel
        prob, pseudo_lbl = torch.max(F.softmax(outputs_tu['out'], dim=1).detach(), dim=1)
        pseudo_lbl_down = F.interpolate(pseudo_lbl.unsqueeze(0).float(), size=(pred_tu.shape[2], pred_tu.shape[3]), mode='nearest').squeeze()
        prob_down = F.interpolate(prob.unsqueeze(0), size=(pred_tu.shape[2], pred_tu.shape[3]), mode='nearest').squeeze()

        # take out the features from black pixels from zooms out and augmetnations 
        ignore_label = 250
        threshold = 0.9
        mask = prob_down > threshold
        mask = mask * (pseudo_lbl_down != ignore_label)    # this is legacy from Alonso et al, but might be useful if we introduce zooms and crops

        pred_tu = pred_tu.permute(0, 2, 3, 1)
        pred_tu = pred_tu[mask, ...]
        pseudo_lbl_down = pseudo_lbl_down[mask]

        loss_unlabeled = contrastive_class_to_class(None, pred_tu, pseudo_lbl_down, self.feature_memory.memory)
        return loss_unlabeled
