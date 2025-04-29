import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from utils import log, colorstr
from utils.peft_utils import load_hf_adapter
from utils.common_utils import replace_none_value
from utils.training_utils import choose_proper_resume_model
from tools import EleutherEvaluator
from trainer.build import get_model, get_peft_model



def env_setup():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    config = Config(config_path)
    config = replace_none_value(config)
    config = Config(config)     # remapping config
    return config


def resume_model(args):
    device = torch.device('cpu')
    config = load_config(os.path.join(args.resume_model_dir, 'args.yaml'))
    
    # Load model checkpoint
    model, tokenizer = get_model(config, device)
    if config.peft_config_path and config.adapter_save_type != 'merge':
        model = get_peft_model(model, config)
        load_hf_adapter(model, args.adapter_path)
        resume_path = args.adapter_path
    else:
        resume_path = choose_proper_resume_model(args.resume_model_dir, args.load_model_type)
        checkpoints = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoints['model'])
    log(f'Resumed model: {colorstr(resume_path)}')

    return model, tokenizer


def main(args):    
    # Init config
    config = load_config(args.config)
    torch.set_num_threads(config.total_cpu_use)
    
    # Init environment
    env_setup()
    
    # Eleuther AI's lm_eval harness evaluation
    write_path = args.results_write_path if args.results_write_path else os.path.join(args.resume_model_dir, 'eleuther_eval_results.json')
    model, tokenizer = resume_model(args)
    evaluator = EleutherEvaluator(config)
    evaluator.model_evaluate(model, tokenizer, write_path=write_path)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume_model_dir', default=None, type=str, required=True)
    parser.add_argument('-a', '--adapter_path', default=None, type=str, required=False)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    parser.add_argument('-o', '--results_write_path', default=None, type=str, required=False)
    args = parser.parse_args()

    main(args)

    