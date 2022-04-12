import torch
import torch.nn.functional as F
import pdb 

def consistency_reg(cr_type, out_w, out_s, tau=0.9):
    if cr_type == 'one_hot':
        return cr_one_hot(out_w, out_s, tau)
    elif cr_type == 'prob_distr':
        return cr_prob_distr(out_w, out_s, tau)
    else:
        raise Exception('Consistancy regularization type not supported')


def cr_one_hot(out_w, out_s, tau):
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
    percent_pl = sum(pseudo_lbl.unique(return_counts=True)[1][:-1]) / len(pseudo_lbl) * 100

    return loss_cr, percent_pl
    

def cr_prob_distr(out_w, out_s, tau, loss='CE'):
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
    idxs = torch.where(max_prob > tau, 1, 0).nonzero().squeeze()
    if idxs.nelement() == 0:  
        return 0, 0

    # Apply only CE (between distributions!) where confidence > threshold    
    out_s = out_s.permute(0, 2, 3, 1)
    out_s = torch.flatten(out_s, end_dim=2)
    out_s = out_s[idxs]
    p_w = p_w[idxs]
    
    if idxs.nelement() == 1: # when a single pixel is above the threshold, need to add a dimension
        idxs = idxs.unsqueeze(0)
        out_s = out_s.unsqueeze(0)
        p_w = p_w.unsqueeze(0)
    assert out_s.size() == p_w.size()

    if idxs.nelement() > 1: # when a single pixel is above the threshold, need to add a dimension
        pdb.set_trace()

    if loss == 'CE':
        loss_cr = F.cross_entropy(out_s, p_w)
    elif loss == 'JS':
        pass #loss_cr = 
    percent_pl = len(idxs) / len(max_prob) * 100

    return loss_cr, percent_pl


