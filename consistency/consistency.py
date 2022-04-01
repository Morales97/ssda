import torch
import torch.nn.functional as F
import pdb 

def cr_one_hot(out_w, out_s, tau=0.9):
    '''
    Consistency regularization with pseudo-labels encoded as One-hot.

    :out_w: Outputs for the batch of weak image augmentations
    :out_s: Outputs for the batch of strong image augmentations
    :tau: Threshold of confidence to use prediction as pseudolabel
    '''
    out_w = out_w.permute(0, 2, 3, 1)         # (N, H, W, C)
    out_w = torch.flatten(out_w, end_dim=2)   # (N·H·W, C)
    p_w = F.softmax(out_w, dim=1).detach()    # compute softmax along classes dimension

    # Generate one-hot pseudo-labels
    max_prob, pseudo_lbl = torch.max(p_w, dim=1)
    pseudo_lbl = torch.where(max_prob > tau, pseudo_lbl, 250)   # 250 is the ignore_index
    # pseudo_lbl.unique(return_counts=True)

    out_s = out_s.permute(0, 2, 3, 1)
    out_s = torch.flatten(out_s, end_dim=2)
    assert len(pseudo_lbl) == out_s.size()[0]

    loss_cr = F.cross_entropy(out_s, pseudo_lbl, ignore_index=250)
    
    if pseudo_lbl.unique(return_counts=True)[0] > 1:
        pdb.set_trace()
    return loss_cr, sum(pseudo_lbl != 250)
    

def cr_prob_distr(out_w, out_s, tau=0.9):
    '''
    Consistency regularization with pseudo-labels encoded as One-hot.

    :out_w: Outputs for the batch of weak image augmentations
    :out_s: Outputs for the batch of strong image augmentations
    :tau: Threshold of confidence to use prediction as pseudolabel
    '''
    out_w = out_w.permute(0, 2, 3, 1)         # (N, H, W, C)
    out_w = torch.flatten(out_w, end_dim=2)   # (N·H·W, C)
    p_w = F.softmax(out_w, dim=1).detach()        

    max_prob, _ = torch.max(p_w, dim=1)
    idxs = torch.where(max_prob > tau, 1, 0).nonzero()

    # Apply only CE (between distributions!) where confidence > threshold
    out_s = out_s[idxs].permute(0, 2, 3, 1)
    out_s = torch.flatten(out_s, end_dim=2)
    p_w = p_w[idxs]
    assert out_s.size() == p_w.size()

    loss_cr = F.cross_entropy(out_s, p_w, ignore_index=250)
    return loss_cr