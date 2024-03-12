import os
import sys
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from trainer import Trainer, Chatter


def env_setup():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):    
    # init config
    config = load_config(args.config)
    config.yaml_file = args.config
    
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
    device = torch.device('cpus') if config.device == False else torch.device(f'cuda:{config.device[0]}')
    trainer = Trainer(config, args.mode, device, use_huggingface_trainer=args.use_huggingface_trainer)

    if args.mode == 'train':
        trainer.do_train()


def multi_gpu_train(gpu, ngpus_per_node, config, args):
    # init distribution
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:10001', world_size=ngpus_per_node, rank=gpu)
    torch.cuda.set_device(gpu)
    torch.distributed.barrier()
    trainer = Trainer(config, args.mode, gpu, is_ddp=True)

    if args.mode == 'train':
        trainer.do_train()


def chat(args):
    config = load_config(os.path.join(args.saved_model_dir, 'args.yaml'))
    model_path = os.path.join(args.saved_model_dir, 'model.pt')
    device = torch.device('cpu') if args.device == 'cpu' else torch.device(f'cuda:{args.device}')
    chatter = Chatter(config, model_path, device)
    chatter.do_chat()



    



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False)
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'chat'])
    parser.add_argument('-p', '--saved_model_dir', type=str, required=False)
    parser.add_argument('-d', '--device', type=str, required=False)
    parser.add_argument('--use_huggingface_trainer', action='store_true')
    args = parser.parse_args()

    if args.mode == 'train':
        assert args.config, 'config file is required for training'
        main(args)
    elif args.mode == 'chat':
        assert args.saved_model_dir, 'saved_model_dir is required for chat'
        assert args.device, 'device is required for chat, you can choose cpu or gpu number like 0,1,2,3'
        chat(args)

    