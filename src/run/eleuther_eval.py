import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch



def env_setup():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):    
    # Init config
    config = load_config(args.config)
    config.yaml_file = args.config
    config.training_stage = args.stage
    
    # Init environment
    env_setup()
    
    # Eleuther AI's lm_eval harness evaluation
    eleuther_evaluation(args, config)

    
def eleuther_evaluation(args, config):
    torch.set_num_threads(config.total_cpu_use)
    device = torch.device('cpu') if config.device == 'cpu' else torch.device(f'cuda:{config.device[0]}')
    if device.type == 'cuda':
        torch.cuda.set_device(config.device[0])



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume_model_dir', default=None, type=str, required=True)
    parser.add_argument('-a', '--adapter_path', default=None, type=str, required=False)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    args = parser.parse_args()

    main(args)

    