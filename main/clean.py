import sys
sys.path.append("../")

import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning, message=r"To copy construct from a tensor, it is recommended to use sourceTensor\.clone\(\)\.detach\(\).*",
)
warnings.filterwarnings( "ignore",category=UserWarning,  message=r"Glyph .* missing from font\(s\) .*",)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import wandb
# import swanlab
# swanlab.sync_wandb()
import argparse
import yaml
import traceback

import torch
import torchvision
import numpy as np
import random

from fl_utils.helper import Helper
from fl_utils.fler import FLer


import os

def setup_wandb(config_path, sweep):
    with open(config_path, 'r', encoding='utf-8') as stream:
        sweep_configuration = yaml.safe_load(stream)

    if sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='FanL-clean')
        return sweep_id
    else:
        #===只提取parameter部分的参数===
        config = sweep_configuration['parameters']
        d = dict()
        for k in config.keys():
            v = config[k][list(config[k].keys())[0]]  
            if type(v) is list:  
                d[k] = {'value':v[0]}
            else:
                d[k] = {'value':v}  
        yaml.dump(d, open('./yamls/tmp.yaml','w', encoding='utf-8'))
        wandb.init(
                project = "Our_method_test",
                name = str(config['attack_type']['value']) + '_' + \
                        str(config['dataset']['value']) + '_' + \
                        str(config['model']['value'])+ '_' + \
                        str(config['agg_method']['value']) + '_' + \
                        str(config['poison_start_epoch']['value']),
                config='./yamls/tmp.yaml',
                mode="offline"
                   ) # ←设置为了offline模式
        return None

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(params):
    run = wandb.init(mode="offline") # if you want use wandb for record, set "mode="online""
    set_seed(wandb.config.seed)
    helper = Helper(wandb.config)
    fler = FLer(helper)
    fler.train() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default = './yamls/poison.yaml')
    parser.add_argument('--gpu', default = 0)
    parser.add_argument('--sweep', action = 'store_true')
    args = parser.parse_args()
    torch.cuda.set_device(int(args.gpu))
    sweep_id = setup_wandb(args.params, args.sweep)

    print("sweep_id:",sweep_id)
    if args.sweep:  
        wandb.agent(sweep_id, function=main, count=1)
    else:
        main(args.params)