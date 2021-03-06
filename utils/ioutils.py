import argparse
import logging
import math
import os
import random
import string
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser(description='SSDA Classification')
    parser.add_argument('--expt_name', type=str, default='',
                        help='Name of the experiment for wandb')
    parser.add_argument('--steps', type=int, default=40000, metavar='N',
                        help='maximum number of iterations to train')
    parser.add_argument('--steps_job', type=int, default=0,
                        help='maximum number of iterations in one slurm job')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay used by optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum used by optimizer')
    parser.add_argument('--backbone_path', type=str, default='',
                        help='Path to checkpoint to load backbone from')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay', type=str, default='det',
                        help='lr decay to use (det/poly')         
    parser.add_argument('--batch_size_s', type=int, default=2,
                        help='Batch size of source labelled data.')
    parser.add_argument('--batch_size_tl', type=int, default=2,
                        help='Batch size of target labelled data.')
    parser.add_argument('--batch_size_tu', type=int, default=2,
                        help='Batch size of target unlabelled data.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker threads to use in each dataloader')
    parser.add_argument('--max_num_threads', type=int, default=12,
                        help='Maximum number of threads that the process should '
                             'use. Uses torch.set_num_threads()')
    parser.add_argument('--clip_norm', type=float, default=10,
                        help='norm for gradient clipping')
    parser.add_argument('--project', type=str, default='GTA_to_CS_tiny',
                        help='wandb project to use')
    parser.add_argument('--entity', type=str, default='morales97',
                        help='wandb entity to use')
    parser.add_argument('--save_dir', type=str, default='expts/tmp_last',
                        help='dir to save experiment results to')
    parser.add_argument('--save_model', type=boolfromstr, default=True,
                        help='If not set, model will not be saved')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--val_interval', type=int, default=250, metavar='N',
                        help='how many batches to wait before validation')
    parser.add_argument('--save_interval', type=int, default=40000, metavar='N',
                        help='how many batches to wait before saving a model')
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint path to resume from')
    parser.add_argument('--pre_trained', type=boolfromstr, default=False,
                        help='Segmentation model pretrained end-to-end, usually on COCO')
    parser.add_argument('--pre_trained_backbone', type=boolfromstr, default=True,
                        help='Backbone of the seg model pretrained, usually on ImageNet')
    parser.add_argument('--net', type=str, default='deeplabv2_rn101',
                        help='choose model architecture')
    parser.add_argument('--target_samples', type=int, default=100,
                        help='how many target domain samples to use. -1: use all samples')
    parser.add_argument('--custom_pretrain', type=str, default=None,
                        help='pretraining. Either denseCL, pixpro, or a custom path')
    parser.add_argument('--cr', type=str, default=None,
                        help='consistency regularization type')
    parser.add_argument('--tau', type=float, default=0.9,
                        help='threshold for pseudolabels')
    parser.add_argument('--lmbda', type=float, default=1,
                        help='weight of consistency regularization loss')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='weight of pixel contrast loss')
    parser.add_argument('--aug_level', type=int, default=5,
                        help='Strong augmentations level')  
    parser.add_argument('--n_augmentations', type=int, default=1,
                        help='number of augmentations in CR')                       
    parser.add_argument('--pixel_contrast', type=boolfromstr, default=False,
                        help='Use pixel contrast regularization')
    parser.add_argument('--pc_mixed', type=boolfromstr, default=False,
                        help='Pixel contrast: mix S and T samples or not')   
    parser.add_argument('--pc_memory', type=boolfromstr, default=False,
                        help='Pixel contrast: use memory or not')   
    parser.add_argument('--pc_ema', type=boolfromstr, default=False,
                        help='Pixel contrast: use EMA teacher to generate anchor features or not')   
    parser.add_argument('--hard_anchor', type=boolfromstr, default=True,
                        help='Pixel contrast: use hard anchor sampling or random anchor sampling') 
    parser.add_argument('--alpha', type=float, default=0.995,
                        help='EMA coefficient')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='number of warm up steps before pixel contrast loss')    
    parser.add_argument('--size', type=str, default='small',
                        help='size of the dataset (tiny/small)')     
    parser.add_argument('--dsbn', type=boolfromstr, default=False,
                        help='Use or not Adaptive Batch Norm')    
    parser.add_argument('--alonso_contrast', type=str, default=None,
                        help='Use or not Alonso et al pixel contrastive learning')    
    parser.add_argument('--lab_color', type=boolfromstr, default=False,
                        help='Transform source images into targets LAB color space') 
    parser.add_argument('--gta_cycada', type=boolfromstr, default=False,
                        help='Use GTA stylized as Cityscapes with CyCADA')  
    parser.add_argument('--wandb', type=boolfromstr, default=True,
                        help='whether or not to use wandb')    
    parser.add_argument('--ent_min', type=boolfromstr, default=False,
                        help='Use entropy minimization')     
    parser.add_argument('--eval_ema', type=boolfromstr, default=False,
                        help='For evaluate.py. Evaluate on student or on EMA teacher')  
    parser.add_argument('--cutmix_sup', type=boolfromstr, default=False,
                        help='Mix S and T labeled data with CutMix')   
    parser.add_argument('--cutmix_cr', type=boolfromstr, default=True,
                        help='Mix T unlabeled data with CutMix')  
    parser.add_argument('--class_weight', type=boolfromstr, default=True,
                        help='Use class weighting in the loss function')      
    parser.add_argument('--pseudolabel_folder', type=str, default=None,
                        help='path to folder with pseudolabels')     
    parser.add_argument('--prev_teacher', type=str, default=None,
                        help='path to teacher model')  
    parser.add_argument('--dropPL_step', type=int, default=-1,
                        help='step when to drop pseudolabels')    
    parser.add_argument('--mixed_batch', type=boolfromstr, default=True,
                        help='Mix S and T data in mini-batch')   
    return parser
    


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args

def get_default_args():
    parser = get_parser()
    args = parser.parse_args([])
    return args

def boolfromstr(s):
    if s.lower().startswith('true'):
        return True
    elif s.lower().startswith('false'):
        return False
    else:
        raise Exception('Incorrect option passed for a boolean')

class FormattedLogItem:
    def __init__(self, item, fmt):
        self.item = item
        self.fmt = fmt
    def __str__(self):
        return self.fmt.format(self.item)

def rm_format(dict_obj):
    ret = dict_obj
    for key in ret:
        if isinstance(ret[key], FormattedLogItem):
            ret[key] = ret[key].item
    return ret

def get_log_str(args, log_info, title='Expt Log', sep_ch='-'):
    now = str(datetime.now().strftime("%H:%M %d-%m-%Y"))
    log_str = (sep_ch * math.ceil((80 - len(title))/2.) + title
               + (sep_ch * ((80 - len(title))//2)) + '\n')
    log_str += '{:<25} : {}\n'.format('Time', now)
    for key in log_info:
        log_str += '{:<25} : {}\n'.format(key, log_info[key])
    log_str += sep_ch * 80
    return log_str

def write_to_log(args, log_str, mode='a+'):
    with open(os.path.join(args.save_dir, 'log.txt'), mode) as outfile:
        print(log_str, file=outfile)

def get_logger(args):
    log_config = {
        'level': logging.INFO,
        'format': '{asctime:s} {levelname:<8s} {filename:<12s} : {message:s}',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'filename': os.path.join(args.save_dir, 'events.log'),
        'filemode': 'w',
        'style': '{'}
    logging.basicConfig(**log_config)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    cfmt = logging.Formatter('{asctime:s} : {message:s}',
                             style='{', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(cfmt)
    logger = logging.getLogger(__name__)
    logger.addHandler(console)

    return logger

def gen_unique_name(length=4):
    """
    Returns a string of 'length' lowercase letters
    """
    return ''.join([random.choice(
        string.ascii_lowercase) for i in range(length)])

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class WandbWrapper():
    def __init__(self, debug=False):
        self.debug = debug
        if debug:
            print('Wandb Wrapper : debug mode. No logging with wandb')

    def init(self, *args, **kwargs):
        if not self.debug:
            import wandb
            wandb.init(*args, **kwargs)
            self.run = wandb.run
        else:
            self.run = AttrDict({'dir' : kwargs['dir']})

    def log(self, *args, **kwargs):
        if not self.debug:
            import wandb
            wandb.log(*args, **kwargs)

    def join(self, *args, **kwargs):
        if not self.debug:
            import wandb
            wandb.join(*args, **kwargs)