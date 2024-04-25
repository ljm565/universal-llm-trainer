import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils import LOGGER, colorstr, TQDM



def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


def choose_proper_model(config):
    pattern = r'\b(\d+.\d+|\d+)b\b'
    target_size = float(config.model_size.lower()[:-1]) \
        if isinstance(config.model_size, str) and config.model_size.lower().endswith('b') \
            else config.model_size

    if config.model.lower() == 'bagel':
        model_list = [
            'jondurbin/nontoxic-bagel-34b-v0.2',
            'jondurbin/bagel-dpo-7b-v0.4'
        ]
        size_diff = [abs(target_size - float(re.findall(pattern, text.lower())[0])) \
                            for text in model_list]
        idx = size_diff.index(min(size_diff))
    elif config.model.lower() == 'koalpaca':
        model_list = [
            'beomi/KoAlpaca-Polyglot-5.8B',
        ]
        size_diff = [abs(target_size - float(re.findall(pattern, text.lower())[0])) \
                            for text in model_list]
        idx = size_diff.index(min(size_diff))
    elif config.model.lower() == 't3q_solar':
        model_list = [
            'chihoonlee10/T3Q-ko-solar-dpo-v3.0-10.7B',
        ]
        size_diff = [abs(target_size - float(re.findall(pattern, text.lower())[0])) \
                            for text in model_list]
        idx = size_diff.index(min(size_diff))
        model_list[idx] = '-'.join(model_list[idx].split('-')[:-1])
    elif config.model.lower() == 'llama3':
        model_list = [
            'meta-llama/Meta-Llama-3-8B-Instruct',
        ]
        size_diff = [abs(target_size - float(re.findall(pattern, text.lower())[0].split('-')[-1])) \
                            for text in model_list]
        idx = size_diff.index(min(size_diff))
    elif config.model.lower() == 'kogemma':
        model_list = [
            'gemmathon/gemma-2b-ko-dev-pbmt192',
        ]
        size_diff = [abs(target_size - float(re.findall(pattern, text.lower())[0])) \
                            for text in model_list]
        idx = size_diff.index(min(size_diff))
    else:
        raise NotImplementedError
    
    # logs
    if config.is_rank_zero:
        LOGGER.info(f"Chosen model: {colorstr(model_list[idx])}")
    return model_list[idx]


def choose_proper_resume_model(resume_dir, type):
    weights_dir = os.listdir(os.path.join(resume_dir, 'weights'))
    try:
        weight = list(filter(lambda x: type in x, weights_dir))[0]
        return os.path.join(resume_dir, 'weights', weight)
    except IndexError:
        raise IndexError(f"There's no model path in {weights_dir} of type {type}")


def draw_training_lr_curve(config, func, all_steps_n, warmup_steps_n, is_ddp, world_size):
    save_dir = os.path.join(config.save_dir, 'vis_data')
    lr0 = config.lr0
    
    os.makedirs(save_dir, exist_ok=True)
    lrs = [func(i)*lr0 if i > warmup_steps_n
           else lr_warmup(i, warmup_steps_n, lr0, func) for i in range(all_steps_n)]
    plt.figure(figsize=(8, 6))
    plt.plot(range(all_steps_n), lrs, marker='o')
    plt.xlabel(f'{config.optimizer_step_criterion}')
    plt.ylabel('Learning Rate')
    if is_ddp:
        plt.title(f'Learning Rate Schedule per GPU (World Size: {world_size})')
    else:
        plt.title('Learning Rate Schedule')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_schedule.png'))


def lr_warmup(cur_step, warmup_steps_n, lr0, func):
    new_lr = np.interp(
        cur_step, 
        [0, warmup_steps_n], 
        [0, lr0*func(cur_step)]
    )
    return new_lr


def init_train_progress_bar(dloader, is_rank_zero, loss_names, nb):
    if is_rank_zero:
        header = tuple(['Epoch', 'GPU_mem'] + \
                loss_names)
        LOGGER.info(('\n' + '%15s' * (2 + len(loss_names))) % header)
        pbar = TQDM(enumerate(dloader), total=nb)
    else:
        pbar = enumerate(dloader)
    return pbar