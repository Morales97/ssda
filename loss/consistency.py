import torch
import torch.nn.functional as F
import pdb 
from collections import OrderedDict
from torchvision.utils import save_image


def consistency_reg(cr_type, out_w, out_s, tau=0.9):
    # Weak augmentations
    out_w = out_w.permute(0, 2, 3, 1)         # (N, H, W, C)
    out_w = torch.flatten(out_w, end_dim=2)   # (N·H·W, C)
    p_w = F.softmax(out_w, dim=1).detach()    # compute softmax along classes dimension

    # Strong augmentations
    out_s = out_s.permute(0, 2, 3, 1)         # (N, H, W, C)
    out_s = torch.flatten(out_s, end_dim=2)   # (N·H·W, C)
    p_s = F.softmax(out_s, dim=1)  

    if cr_type == 'one_hot':
        return cr_one_hot(p_w, out_s, tau)
    elif cr_type == 'prob_distr':
        return cr_prob_distr(p_w, out_s, tau)
    elif cr_type == 'js':
        return cr_JS(p_w, p_s)
    elif cr_type == 'js_th':
        return cr_JS_th(p_w, p_s, tau)
    elif cr_type == 'js_oh':
        return cr_JS_one_hot(p_w, p_s, tau)
    elif cr_type == 'kl':
        return cr_KL(p_w, p_s)
    elif cr_type == 'kl_th':
        return cr_KL_th(p_w, p_s, tau)
    elif cr_type == 'kl_oh':
        return cr_KL_one_hot(p_w, p_s, tau)
    else:
        raise Exception('Consistency regularization type not supported')


def _apply_threshold(p_w, tau):
    max_prob, pseudo_lbl = torch.max(p_w, dim=1)
    idxs = torch.where(max_prob > tau, 1, 0).nonzero().squeeze()    # nonzero() returns the indxs where the array is not zero
    if idxs.nelement() == 0:  
        return None, None
    if idxs.nelement() == 1: # when a single pixel is above the threshold, need to add a dimension
        idxs = idxs.unsqueeze(0)
    return idxs, pseudo_lbl

def _to_one_hot(pseudo_lbl, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pseudo_lbl_oh = torch.zeros((len(pseudo_lbl), n_classes)).to(device)
    pseudo_lbl_oh[:, pseudo_lbl] = 1
    return pseudo_lbl_oh

# *** Cross-Entropy ***
def cr_one_hot(p_w, out_s, tau):
    '''
    Consistency regularization with pseudo-labels encoded as One-hot.
    '''
    # Generate one-hot pseudo-labels
    max_prob, pseudo_lbl = torch.max(p_w, dim=1)
    pseudo_lbl = torch.where(max_prob > tau, pseudo_lbl, 250)   # 250 is the ignore_index
    # pseudo_lbl.unique(return_counts=True)

    assert len(pseudo_lbl) == out_s.size()[0]

    loss_cr = F.cross_entropy(out_s, pseudo_lbl, ignore_index=250)
    percent_pl = sum(pseudo_lbl != 250) / len(pseudo_lbl) * 100
    pdb.set_trace()
    return loss_cr, percent_pl
    
def cr_prob_distr(p_w, out_s, tau):
    '''
    Consistency regularization with pseudo-labels as a probability distribution
    '''
    n = out_s.shape[0]
    idxs, _ = _apply_threshold(p_w, tau)
    if idxs is None: return 0, 0

    out_s = out_s[idxs]
    p_w = p_w[idxs]
    assert out_s.size() == p_w.size()

    loss_cr = F.cross_entropy(out_s, p_w)
    percent_pl = len(idxs) / n * 100
    return loss_cr, percent_pl

# *** Jensen-Shannon ***
def cr_JS(p_s, p_w, eps=1e-8):            

    m = (p_s + p_w)/2
    kl1 = F.kl_div((p_s + eps).log(), m, reduction='batchmean')   
    kl2 = F.kl_div((p_w + eps).log(), m, reduction='batchmean')
    loss_cr = (kl1 + kl2)/2

    return loss_cr, 100

def cr_JS_th(p_s, p_w, tau, eps=1e-8):
    n = p_s.shape[0]
    idxs, _ = _apply_threshold(p_w, tau)
    if idxs is None: return 0, 0

    p_s = p_s[idxs]
    p_w = p_w[idxs]
    assert p_s.size() == p_w.size()

    loss_cr, _ = cr_JS(p_s, p_w)
    percent_pl = len(idxs) / n * 100
    return loss_cr, percent_pl

def cr_JS_one_hot(p_w, p_s, tau, eps=1e-8):    
    n = p_s.shape[0]
    idxs, pseudo_lbl = _apply_threshold(p_w, tau)
    if idxs is None: return 0, 0

    # filter by confidence > tau
    pseudo_lbl = pseudo_lbl[idxs]
    p_s = p_s[idxs]

    # Generate one-hot pseudo-labels
    pseudo_lbl_oh = _to_one_hot(pseudo_lbl, n_classes=p_w.shape[1])

    # compute Jensen-Shannon div
    m = (p_s + pseudo_lbl_oh)/2
    kl1 = F.kl_div((p_s + eps).log(), m, reduction='batchmean')   
    kl2 = F.kl_div((pseudo_lbl_oh + eps).log(), m, reduction='batchmean')
    loss_cr = (kl1 + kl2)/2
    
    percent_pl = len(idxs) / n * 100
    return loss_cr, percent_pl


# *** KL Divergence ***
def cr_KL(p_w, p_s, eps=1e-8):

    kl = F.kl_div((p_s + eps).log(), p_w, reduction='batchmean')   
    loss_cr = kl
    
    return loss_cr, 100

def cr_KL_th(p_s, p_w, tau, eps=1e-8):
    n = p_s.shape[0]
    idxs, _ = _apply_threshold(p_w, tau)
    if idxs is None: return 0, 0

    p_s = p_s[idxs]
    p_w = p_w[idxs]
    assert p_s.size() == p_w.size()

    loss_cr, _ = cr_KL(p_s, p_w)
    percent_pl = len(idxs) / n * 100
    return loss_cr, percent_pl

def cr_KL_one_hot(p_w, p_s, tau=0.9, eps=1e-8):
    n = p_s.shape[0]
    idxs, pseudo_lbl = _apply_threshold(p_w, tau)
    if idxs is None: return 0, 0

    # filter by confidence > tau
    pseudo_lbl = pseudo_lbl[idxs]
    p_s = p_s[idxs]

    # Generate one-hot pseudo-labels
    pseudo_lbl_oh = _to_one_hot(pseudo_lbl, n_classes=p_w.shape[1]) 

    loss_cr = custom_kl_div(p_s.log(), pseudo_lbl_oh)
    percent_pl = len(idxs) / n * 100

    return loss_cr, percent_pl


def custom_kl_div(prediction, target):
    '''
    From https://github.com/ErikEnglesson/GJS/blob/main/losses.py
    
    Applies 'batchnorm' KL divergence. Can be used with one-hot encoded target
    prediction: log-probabilities
    target: probabilities

    '''
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    return output.mean()


if __name__ == '__main__':
    
    batch_size = 3
    vector_dim = 5

    # Generate random input and target distributions
    p_input = torch.rand(size=(batch_size, vector_dim))
    p_target = torch.rand(size=(batch_size, vector_dim))

    p_input = F.softmax(p_input, dim=1)
    p_target = F.softmax(p_target, dim=1)
    print('\np_input:')
    print(p_input)
    print('\np_target:')
    print(p_target)

    kl_all = F.kl_div(p_input.log(), p_target, reduction='batchmean')
    print('\nkl_all:')
    print(kl_all)

    print('\nkl_batch:')
    total_kl = 0
    for i in range(batch_size):
        kl_batch = F.kl_div(p_input[i].log(), p_target[i], reduction='sum') / batch_size    # this is equivalent to reduction='batchmean' with the entire batch
        print(kl_batch)
        total_kl += kl_batch
    print('total_kl:')
    print(total_kl)

    custom_kl = custom_kl_div(p_input.log(), p_target)
    print('custom_kl:')
    print(custom_kl)

    #kl_one_hot = cr_kl_one_hot()
    pdb.set_trace()
