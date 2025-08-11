import os
import sys
import datetime
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
    # Init config
    config = load_config(args.config)
    config.yaml_file = args.config
    config.training_stage = args.stage
    
    # Init environment
    env_setup()
    
    # Training (cpu/single_gpu or multi_gpu)
    if len(config.device) <= 1 or config.device == 'cpu':
        single_gpu_train(args, config)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))
        ngpus_per_node = len(config.device)
        torch.multiprocessing.spawn(multi_gpu_train, nprocs=ngpus_per_node, args=(ngpus_per_node, config, args))

    
def single_gpu_train(args, config):
    torch.set_num_threads(config.total_cpu_use)
    device = torch.device('cpu') if config.device == 'cpu' else torch.device(f'cuda:{config.device[0]}')
    if device.type == 'cuda':
        torch.cuda.set_device(config.device[0])
    trainer = Trainer(
        config, 
        args.mode, 
        device, 
        use_huggingface_trainer=args.use_huggingface_trainer,
        resume_path=choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.resume_model_dir else None,
        adapter_path=args.adapter_path if args.adapter_path else None,
        gpu_test=args.gpu_test,
    )

    if args.mode in ['train', 'resume']:
        trainer.do_train()


def multi_gpu_train(gpu, ngpus_per_node, config, args):
    torch.set_num_threads(config.total_cpu_use // ngpus_per_node)

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
    trainer = Trainer(
        config,
        args.mode,
        gpu,
        multi_gpu_train_type='fsdp' if config.fsdp_train else 'ddp',
        resume_path=choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.resume_model_dir else None,
        adapter_path=args.adapter_path if args.adapter_path else None,
        gpu_test=args.gpu_test
    )

    if args.mode in ['train', 'resume']:
        trainer.do_train()





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False)
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'resume'])
    parser.add_argument('-r', '--resume_model_dir', default=None, type=str, required=False)
    parser.add_argument('-a', '--adapter_path', default=None, type=str, required=False)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    parser.add_argument('-s', '--stage', type=int, default=0, required=False)
    parser.add_argument('-p', '--port', type=str, default='10001', required=False)
    parser.add_argument('--ddp_timeout', type=int, default=86400, required=False)   # 24 hours
    parser.add_argument('--use_huggingface_trainer', action='store_true')
    parser.add_argument('--gpu_test', action='store_true')
    args = parser.parse_args()

    # Sanity checks
    if args.resume_model_dir or args.adapter_path:
        assert args.mode == 'resume', colorstr('red', 'Please set mode to resume..')
    
    if args.mode == 'train':
        assert args.config, colorstr('red', 'config file is required for training..')
        main(args)
    elif args.mode == 'resume':
        assert args.config, colorstr('red', 'config file is required for training..')
        main(args)

    