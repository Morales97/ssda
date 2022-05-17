import numpy as np
import torch
from model.model import get_model
from utils.ioutils import parse_args
import pdb
from torch_ema import ExponentialMovingAverage

if __name__ == '__main__':
    args = parse_args()

    path_1 = 'expts/tmp_last/checkpoint_KL_pc_cw_r3_3.pth.tar'
    path_2 = 'expts/tmp_last/checkpoint_KL_pc_cw_r3_noPL_3.pth.tar' 

    model_1 = get_model(args)
    model_2 = get_model(args)
    model_1.cuda()
    model_2.cuda()

    ema_1 = ExponentialMovingAverage(model.parameters(), decay=0.995)
    ema_1.to(torch.device('cuda'))
    ema_2 = ExponentialMovingAverage(model.parameters(), decay=0.995)
    ema_2.to(torch.device('cuda'))

    checkpoint_1 = torch.load(path_1)
    model_1.load_state_dict(checkpoint_1['model_state_dict'])
    if 'ema_state_dict' in checkpoint_1.keys():
        ema_1.load_state_dict(checkpoint_1['ema_state_dict'])

    checkpoint_2 = torch.load(path_2)
    model_2.load_state_dict(checkpoint_2['model_state_dict'])
    if 'ema_state_dict' in checkpoint_2.keys():
        ema_2.load_state_dict(checkpoint_2['ema_state_dict'])

    pdb.set_trace()