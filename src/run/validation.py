import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from utils import colorstr
from utils.training_utils import choose_proper_resume_model
from trainer import Trainer


def env_setup():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):    
    # init config
    config = load_config(os.path.join(args.resume_model_dir, 'args.yaml'))
    
    # init environment
    env_setup()
    
    # training (cpu/single_gpu or multi_gpu)
    validation(args, config)

    
def validation(args, config):
    device = torch.device('cpus') if config.device == False else torch.device(f'cuda:{config.device[0]}')
    trainer = Trainer(
        config, 
        'validation', 
        device, 
        resume_path=choose_proper_resume_model(args.resume_model_dir, args.load_model_type)
    )

    trainer.epoch_validate('validation', 0, False)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--resume_model_dir', type=str, required=False)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    args = parser.parse_args()

    main(args)

    