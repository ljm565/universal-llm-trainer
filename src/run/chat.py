import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from tools import Chatter



def load_config(config_path):
    config = Config(config_path)
    return config


def chat(args):
    model_dir = args.saved_model_dir
    config = Config(os.path.join(model_dir, 'args.yaml'))
    model_name = list(filter(lambda x: args.load_model_type in x, os.listdir(os.path.join(model_dir, 'weights'))))[0]
    model_path = os.path.join(model_dir, 'weights', model_name)
    device = torch.device('cpu') if args.device == 'cpu' else torch.device(f'cuda:{args.device}')

    chatter = Chatter(
        config, 
        model_path, 
        device,
        args.template_path,
        args.is_multi_turn,
        args.efficient_load
    )

    chatter.do_chat(args.is_greedy)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--saved_model_dir', type=str, required=True)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    parser.add_argument('-d', '--device', type=str, default='0', required=False)
    parser.add_argument('-t', '--template_path', type=str, required=True)
    parser.add_argument('-g', '--is_greedy', action='store_true', required=False)
    parser.add_argument('-m', '--is_multi_turn', action='store_true', required=False)
    parser.add_argument('-e', '--efficient_load', action='store_true', required=False)
    args = parser.parse_args()

    chat(args)
    


    