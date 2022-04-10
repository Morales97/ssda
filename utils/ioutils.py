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
    parser.add_argument('--steps', type=int, default=5000, metavar='N',
                        help='maximum number of iterations to train')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay used by optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum used by optimizer')
    parser.add_argument('--backbone_path', type=str, default='',
                        help='Path to checkpoint to load backbone from')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--batch_size_s', type=int, default=8,
                        help='Batch size of source labelled data.')
    parser.add_argument('--batch_size_tl', type=int, default=8,
                        help='Batch size of target labelled data.')
    parser.add_argument('--batch_size_tu', type=int, default=8,
                        help='Batch size of target unlabelled data.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker threads to use in each dataloader')
    parser.add_argument('--max_num_threads', type=int, default=12,
                        help='Maximum number of threads that the process should '
                             'use. Uses torch.set_num_threads()')

    parser.add_argument('--project', type=str, default='GTA_to_CS_tiny',
                        help='wandb project to use')
    parser.add_argument('--entity', type=str, default='morales97',
                        help='wandb entity to use')
    parser.add_argument('--save_dir', type=str, default='expts/tmp_last',
                        help='dir to save experiment results to')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='If not set, model will not be saved')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--val_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before validation')
    parser.add_argument('--save_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before saving a model')
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint path to resume from')
    parser.add_argument('--pre_trained', type=boolfromstr, default=False,
                        help='Segmentation model pretrained end-to-end, usually on COCO')
    parser.add_argument('--pre_trained_backbone', type=boolfromstr, default=True,
                        help='Backbone of the seg model pretrained, usually on ImageNet')
    parser.add_argument('--net', type=str, default='lraspp_mobilenet',
                        help='choose model architecture')
    parser.add_argument('--target_samples', type=int, default=100,
                        help='how many target domain samples to use. Default -1: use all samples')
    parser.add_argument('--custom_pretrain_path', type=str, default=None,
                        help='path to load pretrained model from')
    parser.add_argument('--cr', type=str, default=None,
                        help='consistency regularization type')
    parser.add_argument('--lmbda', type=int, default=1,
                        help='weight of consistency regularization loss')
    parser.add_argument('--pixel_contrast', type=boolfromstr, default=False,
                        help='Use pixel contrast regularization')
    parser.add_argument('--gamma', type=int, default=1,
                        help='weight of pixel contrast loss')
    parser.add_argument('--warmup_steps', type=int, default=2500,
                        help='number of warm up steps before pixel contrast loss')              
    '''
    parser.add_argument('--net', type=str, default='resnet34',
                        choices=['alexnet', 'vgg', 'resnet34'],
                        help='which network to use')
    parser.add_argument('--source', type=str, default='real',
                        help='source domain')
    parser.add_argument('--target', type=str, default='sketch',
                        help='target domain')
    parser.add_argument('--dataset', type=str, default='multi',
                        choices=['multi', 'office', 'office_home', 'visda17'],
                        help='the name of dataset')
    parser.add_argument('--num', type=int, default=3,
                        help='number of labeled examples in the target')

    parser.add_argument('--aug_level', type=int, default=3,
                        help='Level of augmentation to apply to data. Currently:'
                             '0 : Resize, randomcrop, random flip'
                             '1 : 0 + color jitter '
                             '2 : 0 + Randaugment'
                             '3 : Randaugment + Color jittering'
                             '4 : 3 with lower rotation and sheer')

    parser.add_argument('--fs_ss', action='store_true', default=False,
                        help='Whether to use file_system as '
                             'torch.mutiprocessing sharing strategy')
    '''
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