import torch
import torch.nn.functional as F

def entropy_loss(out):
    n, c, h, w = out.size()

    prob = F.softmax(out, dim=1)
    log_prob = torch.log(prob + 1e-10)
    entropy = - prob * log_prob
    entropy = torch.sum(entropy.view(-1)) / (n*h*w)

    return entropy