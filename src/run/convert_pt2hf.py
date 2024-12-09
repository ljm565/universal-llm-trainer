import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from utils import LOGGER, colorstr
from utils.func_utils import replace_none_value
from utils.training_utils import choose_proper_resume_model
from trainer.build import get_model


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
    # init config
    config = load_config(os.path.join(args.resume_model_dir, 'args.yaml')) if args.resume_model_dir else load_config(args.config)
    if 'training_stage' not in config:
        config.training_stage = 0
    
    # init environment
    env_setup()
    
    # validation
    resume_model(args, config)

    
def resume_model(args, config):
    torch.set_num_threads(config.total_cpu_use)
    
    device = torch.device('cpu')
    resume_path = choose_proper_resume_model(args.resume_model_dir, args.load_model_type)
    
    # Load model checkpoint
    checkpoints = torch.load(resume_path, map_location=device)
    model, tokenizer = get_model(config, device)
    model.load_state_dict(checkpoints['model'])
    LOGGER.info(f'Resumed model: {colorstr(resume_path)}')

    # Extract the only Hugging Face model
    model.model.save_pretrained(args.save_model_dir)
    tokenizer.tokenizer.save_pretrained(args.save_model_dir)
    LOGGER.info(f'Conversion is completed: {colorstr(args.save_model_dir)}')

   

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--resume_model_dir', type=str, required=True)
    parser.add_argument('-w', '--save_model_dir', type=str, required=True)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    args = parser.parse_args()

    main(args)

    