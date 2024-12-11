import os
import sys
import datetime
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from utils import colorstr
from utils.training_utils import choose_proper_resume_model
from trainer import LoRoTrainer


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
    config.training_stage = 0
    config.loro_config_path = args.loro_config
    
    # init environment
    env_setup()
    
    # training (cpu/single_gpu or multi_gpu)
    if len(config.device) <= 1 or config.device == 'cpu':
        single_gpu_train(args, config)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))
        ngpus_per_node = len(config.device)
        torch.multiprocessing.spawn(multi_gpu_train, nprocs=ngpus_per_node, args=(ngpus_per_node, config, args))

    
def single_gpu_train(args, config):
    torch.set_num_threads(config.total_cpu_use)

    # Init model to be resumed
    resume_path = {'base_model': choose_proper_resume_model(args.resume_model_dir, args.load_model_type),
                   'model': choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.mode == 'resume' else None}

    device = torch.device('cpu') if config.device == 'cpu' else torch.device(f'cuda:{config.device[0]}')
    if device.type == 'cuda':
        torch.cuda.set_device(config.device[0])
    trainer = LoRoTrainer(
        config, 
        args.mode, 
        device, 
        resume_path=resume_path
    )

    if args.mode in ['train', 'resume']:
        trainer.do_train()


def multi_gpu_train(gpu, ngpus_per_node, config, args):
    torch.set_num_threads(config.total_cpu_use // ngpus_per_node)

    # Init model to be resumed
    resume_path = {'base_model': choose_proper_resume_model(args.resume_model_dir, args.load_model_type),
                   'model': choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.mode == 'resume' else None}

    # Init distribution
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}', 
        world_size=ngpus_per_node,
        rank=gpu, 
        timeout=datetime.timedelta(seconds=args.ddp_timeout)
    )
    torch.cuda.set_device(gpu)
    torch.distributed.barrier()
    trainer = LoRoTrainer(
        config,
        args.mode,
        gpu,
        multi_gpu_train_type='fsdp' if config.fsdp_train else 'ddp',
        resume_path=resume_path
    )

    if args.mode in ['train', 'resume']:
        trainer.do_train()





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--loro_config', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'resume'])
    parser.add_argument('--resume_base_model_dir', type=str, required=True)
    parser.add_argument('-r', '--resume_model_dir', type=str, required=False)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    parser.add_argument('-p', '--port', type=str, default='10001', required=False)
    parser.add_argument('--ddp_timeout', type=int, default=86400, required=False)           # 24 hours
    args = parser.parse_args()

    if args.mode == 'resume':
        assert args.resume_model_dir, colorstr('red', 'Path for model resuming is required')
    
    main(args)

    