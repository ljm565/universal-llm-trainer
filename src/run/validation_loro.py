import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from utils.func_utils import replace_none_value
from utils.training_utils import choose_proper_resume_model
from trainer import LoRoTrainer


def env_setup():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    config = Config(config_path)
    config = replace_none_value(config)
    config = Config(config)     # remapping config
    return config


def main(args):    
    # Init config
    config = load_config(os.path.join(args.resume_model_dir, 'args.yaml'))
    config.test_rejection_data = True if args.test_rejection_data else False
    
    # Init environment
    env_setup()
    
    # Validation
    validation(args, config)

    
def validation(args, config):
    torch.set_num_threads(config.total_cpu_use)

    # Init model to be resumed
    resume_path = {'base_model': choose_proper_resume_model(args.resume_base_model_dir, 'metric'),
                   'model': choose_proper_resume_model(args.resume_model_dir, args.load_model_type)}

    if not args.device == None:
        device = torch.device('cpu') if args.device == 'cpu' else torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu') if config.device == 'cpu' else torch.device(f'cuda:{config.device[0]}')
    trainer = LoRoTrainer(
        config, 
        'validation', 
        device, 
        resume_path=resume_path
    )

    trainer.epoch_validate('validation', 0, False)






if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--resume_base_model_dir', type=str, required=True)
    parser.add_argument('-r', '--resume_model_dir', type=str, required=True)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    parser.add_argument('-d', '--device', default=None, required=False)
    parser.add_argument('--test_rejection_data', action='store_true')
    args = parser.parse_args()

    main(args)

    