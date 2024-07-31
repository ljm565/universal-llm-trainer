import os
import sys
import datetime
import deepspeed
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from utils import colorstr
from utils.training_utils import choose_proper_resume_model
from trainer import TrainerDeepSpeed


def env_setup():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):    
    # init config
    config = load_config(args.config)
    config.yaml_file = args.config
    config.training_stage = args.stage
    
    # init environment
    env_setup()
    
    # training (cpu/single_gpu or multi_gpu)
    if len(config.device) <= 1:
        single_gpu_train(args, config)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))
        ngpus_per_node = len(config.device)
        torch.multiprocessing.spawn(multi_gpu_train, nprocs=ngpus_per_node, args=(ngpus_per_node, config, args))

    
def single_gpu_train(args, config):
    device = torch.device('cpu') if config.device == False else torch.device(f'cuda:{config.device[0]}')
    trainer = TrainerDeepSpeed(
        config, 
        args, 
        device, 
        resume_path=choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.mode == 'resume' else None
    )

    if args.mode in ['train', 'resume']:
        trainer.do_train()


def multi_gpu_train(gpu, ngpus_per_node, config, args):
    # init distribution
    deepspeed.init_distributed(backend='nccl', init_method='tcp://127.0.0.1:10001', world_size=ngpus_per_node, rank=gpu, timeout=datetime.timedelta(seconds=args.ddp_timeout))
    torch.cuda.set_device(gpu)
    trainer = TrainerDeepSpeed(
        config,
        args,
        gpu,
        is_ddp=True,
        resume_path=choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.mode == 'resume' else None
    )

    if args.mode in ['train', 'resume']:
        trainer.do_train()





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False)
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'resume'])
    parser.add_argument('-r', '--resume_model_dir', type=str, required=False)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    parser.add_argument('-s', '--stage', type=int, default=0, required=False)
    parser.add_argument('-p', '--port', type=str, default='10001', required=False)
    parser.add_argument('--ddp_timeout', type=int, default=86400, required=False)       # 24 hours               
    parser.add_argument('--use_huggingface_trainer', action='store_true')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    if args.mode == 'train':
        assert args.config, colorstr('red', 'config file is required for training')
        main(args)
    elif args.mode == 'resume':
        assert args.config, colorstr('red', 'config file is required for training')
        assert args.resume_model_dir, colorstr('red', 'Path for model resuming is required')
        main(args)

    