import numpy as np
import torch
from model.model import get_model
from utils.ioutils import parse_args, get_log_str
import pdb
from torch_ema import ExponentialMovingAverage
from evaluation.metrics import averageMeter, runningScore

def _log_validation_ema(model, ema, val_loader, loss_fn, step, wandb=None):
    running_metrics_val = runningScore(19)
    val_loss_meter = averageMeter()
    
    # NOTE DO NOT use model.eval() -> It seems to inhibit the ema.average_parameters() context 
    with ema.average_parameters() and torch.no_grad():
        for (images_val, labels_val) in val_loader:
            images_val = images_val.cuda()
            labels_val = labels_val.cuda()

            outputs = model(images_val)
            outputs = outputs['out']

            val_loss = loss_fn(input=outputs, target=labels_val)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            running_metrics_val.update(gt, pred)
            val_loss_meter.update(val_loss.item())
    
    score, class_iou = running_metrics_val.get_scores()

    log_info = OrderedDict({
        'Train Step': step,
        'Validation loss on EMA': val_loss_meter.avg,
        'mIoU on EMA': score['mIoU'],
        'Overall acc on EMA': score['Overall Acc'],
    })

    log_str = get_log_str(args, log_info, title='Validation Log on EMA')
    print(log_str)

if __name__ == '__main__':
    args = parse_args()

    #_, _, _, val_loader = get_loaders(args)

    path_1 = 'expts/tmp_last/checkpoint_KL_pc_cw_r3_3.pth.tar'
    path_2 = 'expts/tmp_last/checkpoint_KL_pc_cw_r3_noPL_3.pth.tar' 

    model_1 = get_model(args)
    model_2 = get_model(args)
    model_1.cuda()
    model_2.cuda()

    ema_1 = ExponentialMovingAverage(model_1.parameters(), decay=0.995)
    ema_1.to(torch.device('cuda'))
    ema_2 = ExponentialMovingAverage(model_2.parameters(), decay=0.995)
    ema_2.to(torch.device('cuda'))

    checkpoint_1 = torch.load(path_1)
    model_1.load_state_dict(checkpoint_1['model_state_dict'])
    if 'ema_state_dict' in checkpoint_1.keys():
        ema_1.load_state_dict(checkpoint_1['ema_state_dict'])

    checkpoint_2 = torch.load(path_2)
    model_2.load_state_dict(checkpoint_2['model_state_dict'])
    if 'ema_state_dict' in checkpoint_2.keys():
        ema_2.load_state_dict(checkpoint_2['ema_state_dict'])

    ensemble_params = [(p1 + p2)/2 for p1, p2 in zip(ema_1._get_parameters(None), ema_2._get_parameters(None))]
    pdb.set_trace()