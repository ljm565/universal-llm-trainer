import os
import re
import math
import functools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp.wrap import wrap, enable_wrap
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from utils import LOGGER, colorstr, TQDM
from utils.func_utils import wrap_modules


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
    elif config.model.lower() == 'kopolyglot':
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
    elif config.model.lower() == 'gemma':
        model_list = [
            'google/gemma-2b',
            'google/gemma-7b',
        ]
        size_diff = [abs(target_size - float(re.findall(pattern, text.lower())[0])) \
                            for text in model_list]
        idx = size_diff.index(min(size_diff))
    elif config.model.lower() == 'phi3':
        pattern = r'\b(\d+.\d+|\d+)k\b'
        model_list = [
            'microsoft/Phi-3-mini-128k-instruct',
            'microsoft/Phi-3-medium-4k-instruct'
        ]
        if target_size >= 10:
            idx = 1
        else:
            idx = 0
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


def gather_objects(objs, is_rank_zero, world_size):
    gather_list = [None for _ in range(world_size)] if is_rank_zero else None
    dist.gather_object(objs, gather_list, dst=0)
    return gather_list


def calculate_gathered_results(objs):
    # assert whether all sublists have the same keys
    first_keys = list(objs[0]['results'].keys())
    assert all([list(obj['results'].keys()) == first_keys for obj in objs])

    gathered_results = {}
    total_n = sum([obj['length'] for obj in objs])
    for key in first_keys:
        gathered_results[key] = sum(obj['results'][key] * obj['length'] for obj in objs) / total_n
    
    return gathered_results


def get_wrap_policy(config):
    params = config.fsdp_hyperparameters
    wrap_policy = params.wrap_policy
    if wrap_policy.lower() == 'size_based':
        return functools.partial(size_based_auto_wrap_policy, min_num_params=params.size_based.min_num_params)
        # return functools.partial(custom_auto_wrap_policy, is_rank_zero=config.is_rank_zero)
    elif wrap_policy.lower() == 'transformer_based':
        return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_modules(params)})
    else:
        raise NotImplementedError
    

def custom_wrap_policy(model, device):
    def _get_leaf_modules(model):
        leaf_modules = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                leaf_modules.append((name, module))
        return leaf_modules
    
    def _get_parent_module(model, path):
        parts = path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]
  
    leaf_modules = _get_leaf_modules(model)
    for name, module in leaf_modules:
        for param in module.parameters():
            if param.dtype == torch.float32 and param.requires_grad:
                print(f"Wrapping module: {name} of type {module} (dtype: {param.dtype})")
                parent, last_name = _get_parent_module(model, name)
                setattr(parent, last_name, FSDP(module, device_id=device))
                break       # due to leaf modules
    
    return model
    




# def custom_auto_wrap_policy(module, is_rank_zero, recurse, nonwrapped_numel, min_num_params=1e6):
#     # LOGGER.info(colorstr('Custom wrap policy is applied to FSDP training'))
#     # 양자화된 int8 파라미터는 래핑하지 않음
    
#     # print(is_rank_zero)
#     # if any(param.dtype in [torch.int8] for param in module.parameters(recurse=False)):
#     #     return False  # int8 파라미터를 가진 모듈은 FSDP로 감싸지 않음
    
    
    
#     # # 모듈이 파라미터를 가지고 있는지 먼저 확인
#     # if not len(list(module.parameters(recurse=False))):
#     #     return False  # 파라미터가 없는 모듈은 래핑하지 않음

#     # int8 타입의 파라미터가 있는지 확인하여 해당 모듈 제외
#     # int8 타입의 파라미터가 있는 모듈은 완전히 제외
#     for param in module.parameters(recurse=False):
#         if param.dtype != torch.float32:
#             return False  # int8 파라미터가 있으면 래핑하지 않음
    
#     print(f'{is_rank_zero}: {module}\n ------------------------')
#     return True