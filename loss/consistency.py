import torch
import torch.nn.functional as F
import pdb 
from collections import OrderedDict
from torchvision.utils import save_image


def cr_multiple_augs(args, images, model):
    # NOTE for the moment only support 2 augmentations
    assert args.n_augmentations == 2 and args.cr == 'js'
    images_weak = images[0].cuda()
    images_strong1 = images[1].cuda()
    images_strong2 = images[2].cuda()

    outputs_w = model(images_weak)                   # (N, C, H, W)
    outputs_strong1 = model(images_strong1)
    outputs_strong2 = model(images_strong2)
    if type(outputs_w) == OrderedDict:
        out_w = outputs_w['out']
        out_strong1 = outputs_strong1['out']
        out_strong2 = outputs_strong2['out']
    else:
        out_w = outputs_w
        out_strong1 = outputs_strong1
        out_strong2 = outputs_strong2

    return cr_JS_2_augs(out_w, out_strong1, out_strong2)


def consistency_reg(cr_type, out_w, out_s, tau=0.9):
    if cr_type == 'one_hot':
        return cr_one_hot(out_w, out_s, tau)
    elif cr_type == 'prob_distr':
        return cr_prob_distr(out_w, out_s, tau)
    elif cr_type == 'js':
        return cr_JS(out_w, out_s, tau=0)
    elif cr_type == 'kl':
        return cr_KL(out_w, out_s)
    else:
        raise Exception('Consistency regularization type not supported')


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
    

def cr_prob_distr(out_w, out_s, tau):
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


    loss_cr = F.cross_entropy(out_s, p_w)
    
    percent_pl = len(idxs) / len(max_prob) * 100
    return loss_cr, percent_pl


def cr_JS(out_w, out_s, tau, eps=1e-8):
    '''
    TODO generalize to n augmentations
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

    #if idxs.nelement() > 1: # when a single pixel is above the threshold, need to add a dimension
    #    pdb.set_trace()

    out_s = F.softmax(out_s, dim=1)    # convert to probabilities
    m = (out_s + p_w)/2
    kl1 = F.kl_div((out_s + eps).log(), m, reduction='batchmean')   
    kl2 = F.kl_div((p_w + eps).log(), m, reduction='batchmean')
    loss_cr = (kl1 + kl2)/2
    
    percent_pl = len(idxs) / len(max_prob) * 100
    return loss_cr, percent_pl



def cr_JS_2_augs(out_w, out_s1, out_s2, tau=0, eps=1e-8):
    # NOTE only implented for tau = 0
    assert tau == 0

    out_w = out_w.permute(0, 2, 3, 1)         # (N, H, W, C)
    out_w = torch.flatten(out_w, end_dim=2)   # (N·H·W, C)
    out_s1 = out_s1.permute(0, 2, 3, 1)
    out_s1 = torch.flatten(out_s1, end_dim=2)
    out_s2 = out_s2.permute(0, 2, 3, 1)
    out_s2 = torch.flatten(out_s2, end_dim=2)

    p_w = F.softmax(out_w, dim=1).detach()      # stop gradient in original example   
    p_s1 = F.softmax(out_s1, dim=1)    
    p_s2 = F.softmax(out_s2, dim=1)   

    m = (p_w + p_s1 + p_s2)/3
    kl1 = F.kl_div((p_w + eps).log(), m, reduction='batchmean')   
    kl2 = F.kl_div((p_s1 + eps).log(), m, reduction='batchmean')
    kl3 = F.kl_div((p_s2 + eps).log(), m, reduction='batchmean')
    loss_cr = (kl1 + kl2 + kl3)/3
    
    percent_pl = 100
    return loss_cr, percent_pl


def cr_KL(out_w, out_s, eps=1e-8):
    '''
    TODO generalize to n augmentations
    '''
    out_w = out_w.permute(0, 2, 3, 1)         # (N, H, W, C)
    out_w = torch.flatten(out_w, end_dim=2)   # (N·H·W, C)
    out_s = out_s.permute(0, 2, 3, 1)
    out_s = torch.flatten(out_s, end_dim=2)

    p_w = F.softmax(out_w, dim=1).detach()              
    p_s = F.softmax(out_s, dim=1)    

    kl = F.kl_div((p_s + eps).log(), p_w, reduction='batchmean')   
    loss_cr = kl
    
    percent_pl = 100
    return loss_cr, percent_pl

def cr_kl_one_hot(out_w, out_s, tau=0.9, eps=1e-8):
    out_w = out_w.permute(0, 2, 3, 1)         # (N, H, W, C)
    out_w = torch.flatten(out_w, end_dim=2)   # (N·H·W, C)
    p_w = F.softmax(out_w, dim=1).detach()    # compute softmax along classes dimension

    # Generate one-hot pseudo-labels
    max_prob, pseudo_lbl = torch.max(p_w, dim=1)
    pseudo_lbl = torch.where(max_prob > tau, pseudo_lbl+1, 0).nonzero() # +1 to remove 0s without affecting labels 0
    if pseudo_lbl.nelement() == 0:  
        return 0, 0
    pseudo_lbl = pseudo_lbl - 1     # compensate for the +1

    out_s = out_s.permute(0, 2, 3, 1)
    out_s = torch.flatten(out_s, end_dim=2)
    loss_cr = custom_kl_div(out_s.log(), pseudo_lbl)
    return loss_cr

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
