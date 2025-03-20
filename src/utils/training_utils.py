import os
import re
import math
import functools
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, List, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from utils import LOGGER, colorstr, TQDM
from utils.func_utils import wrap_modules
from utils.quant_utils import init_quant_config



def one_cycle(y1:float=0.0, y2:float=1.0, steps:int=100) -> Callable:
    """
    Returns a lambda function for sinusoidal ramp from y1 to y2 as described in the One-Cycle Learning Rate Schedule.
    This function generates a smooth transition between the two values over the specified number of steps.

    Args:
        y1 (float, optional): The starting value of the ramp. Default is 0.0.
        y2 (float, optional): The ending value of the ramp. Default is 1.0.
        steps (int, optional): The number of steps over which the transition occurs. Default is 100.

    Returns:
        Callable[[float], float]: A lambda function that takes a step number and returns the corresponding value in the sinusoidal ramp.
        
    Notes:
        - Reference: https://arxiv.org/pdf/1812.01187.pdf
    """
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1



def is_parallel(model: nn.Module) -> bool:
    """
    Returns True if the model is wrapped in DataParallel (DP) or DistributedDataParallel (DDP).

    Args:
        model (nn.Module): The model to check for parallelism.

    Returns:
        bool: True if the model is in parallel (DP or DDP), False otherwise.
    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))



def de_parallel(model: nn.Module) -> nn.Module:
    """
    De-parallelizes a model by returning the underlying model when it's wrapped in DataParallel (DP) or DistributedDataParallel (DDP).
    Otherwise, returns the model as is.

    Args:
        model (nn.Module): The model to de-parallelize.

    Returns:
        nn.Module: The de-parallelized model.
    """
    return model.module if is_parallel(model) else model



def choose_proper_model(config) -> str:
    """
    Chooses the appropriate model from a predefined list based on the given configuration.
    The model selection is based on the specified model size and model type (e.g., 'llama3', 'gemma', 'phi3').

    Args:
        config: Configuration object that contains the following attributes:
            - model_size (Union[str, float]): The target model size (e.g., '8B', 8).
            - model (str): The name of the model family (e.g., 'kopolyglot', 'llama3', 'gemma').
            - is_rank_zero (bool): Flag indicating whether this is the rank 0 process in distributed training.

    Returns:
        str: The name of the selected model based on the configuration.

    Raises:
        NotImplementedError: If an unsupported model type is specified in the configuration.
    """
    pattern = r'\b(\d+.\d+|\d+)b\b'
    target_size = float(config.model_size.lower()[:-1]) \
        if isinstance(config.model_size, str) and config.model_size.lower().endswith('b') \
            else config.model_size

    if config.model.lower() == 'kopolyglot':
        model_list = [
            'beomi/KoAlpaca-Polyglot-5.8B',
        ]
        size_diff = [abs(target_size - float(re.findall(pattern, text.lower())[0])) \
                            for text in model_list]
        idx = size_diff.index(min(size_diff))
    
    elif config.model.lower() in ['llama3', 'llama3.1']:
        model_list_3 = [
            'meta-llama/Meta-Llama-3-8B-Instruct',
        ]
        model_list_3_1 = [
            'meta-llama/Llama-3.1-8B-Instruct'
        ]
        if config.model.lower() == 'llama3':
            model_list = model_list_3 
        elif config.model.lower() == 'llama3.1':
            model_list = model_list_3_1
        size_diff = [abs(target_size - float(re.findall(pattern, text.lower())[0].split('-')[-1])) \
                            for text in model_list]
        idx = size_diff.index(min(size_diff))
    
    elif config.model.lower() == 'llama2':
        model_list = [
            'meta-llama/Llama-2-13b-hf',
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
    
    elif config.model.lower() in ['gemma', 'gemma1', 'gemma2', 'gemma3']:
        model_list_1 = [
            'google/gemma-2b',
            'google/gemma-7b',
        ]
        model_list_2 = [
            'google/gemma-2-9b-it',
        ]
        model_list_3 = [
            'google/gemma-3-12b-it',
        ]
        if config.model.lower() in ['gemma', 'gemma1']:
            model_list = model_list_1
        elif config.model.lower() == 'gemma2':
            model_list = model_list_2
        elif config.model.lower() == 'gemma3':
            model_list = model_list_3
        size_diff = [abs(target_size - float(re.findall(pattern, text.lower())[0].split('-')[-1])) \
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



def choose_proper_resume_model(resume_dir: str, type: str) -> str:
    """
    Choose the proper model path when model resuming is needed.

    Args:
        resume_dir (str): The directory path for resuming the model.
        type (str): The type of model to resume.

    Returns:
        str: Model weight path.

    Raises:
        IndexError: Raise an error when there's no model path in the directory of 'type'.
    """
    weights_dir = os.listdir(os.path.join(resume_dir, 'weights'))
    try:
        weight = list(filter(lambda x: type in x, weights_dir))[0]
        return os.path.join(resume_dir, 'weights', weight)
    except IndexError:
        raise IndexError(f"There's no model path in {weights_dir} of type {type}")



def draw_training_lr_curve(config, 
                           func: Callable, 
                           all_steps_n: int, 
                           warmup_steps_n: int, 
                           is_ddp: bool, 
                           world_size: int) -> None:
    """
    Draws and saves a learning rate schedule curve.

    Args:
        config: Configuration object containing training settings.
        func (Callable): A callable function for modifying the learning rate.
        all_steps_n (int): Total number of steps in the training process.
        warmup_steps_n (int): Number of steps used for warmup.
        is_ddp (bool): Whether Distributed Data Parallel (DDP) training is used.
        world_size (int): The number of processes involved in DDP training.
    """
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



def lr_warmup(cur_step: int, 
              warmup_steps_n: int, 
              lr0: float, 
              func: Callable) -> float:
    """
    Calculates the learning rate during the warmup phase.


    Args:
        cur_step (int): The current step in training.
        warmup_steps_n (int): The total number of warmup steps.
        lr0 (float): The base learning rate.
        func (Callable): A callable function that modifies the learning rate.

    Returns:
        float: The learning rate at the current step.
    """
    new_lr = np.interp(
        cur_step, 
        [0, warmup_steps_n], 
        [0, lr0*func(cur_step)]
    )
    return new_lr



def init_train_progress_bar(dloader,
                            is_rank_zero: bool,
                            loss_names: List[str],
                            nb: int,
                            interval:str='%15s') -> Union[TQDM, enumerate]:
    """
    Initializes a progress bar for training.

    Args:
        dloader: The data loader for training.
        is_rank_zero (bool): Flag indicating if the current process is the rank 0 process.
        loss_names (List[str]): List of loss names to display in the progress bar.
        nb (int): The total number of dataloader.
        interval (str, optional): Formatting string for the output. Defaults to '%15s'.

    Returns:
        Union[TQDM, enumerate]: The progress bar or an enumerator based on the rank.
    """
    if is_rank_zero:
        header = tuple(['Epoch', 'GPU_mem'] + loss_names)
        LOGGER.info(('\n' + interval * (2 + len(loss_names))) % header)
        pbar = TQDM(enumerate(dloader), total=nb)
    else:
        pbar = enumerate(dloader)
    return pbar



def gather_objects(objs: dict, is_rank_zero: bool, world_size: int) -> List[Any]:
    """
    Gathers objects from multiple processes in a distributed setting.

    Args:
        objs (Any): An object to be gathered.
        is_rank_zero (bool): Flag indicating if the current process is the rank 0 process.
        world_size (int): The total number of processes in the distributed setup.

    Returns:
        List[Any]: The gathered list of objects in rank 0, None in other ranks.
    """
    gather_list = [None for _ in range(world_size)] if is_rank_zero else None
    dist.gather_object(objs, gather_list, dst=0)
    return gather_list



def broadcast_objects(obj: Any, is_rank_zero: bool) -> Any:
    """
    Broadcasts an object from rank 0 to all other ranks in a distributed setup.

    Args:
        obj (Any): The object to be broadcast.
        is_rank_zero (bool): Flag indicating if the current process is the rank 0 process.

    Returns:
        Any: The broadcasted object.
    """
    broadcast_list = [obj] if is_rank_zero else [None]
    dist.broadcast_object_list(broadcast_list, src=0)
    return broadcast_list[0]



def calculate_gathered_results(objs: List[dict]) -> dict:
    """
    Calculates the aggregated results from multiple gathered objects.

    Args:
        objs: A list of gathered objects containing 'results' and 'length' keys.

    Returns:
        dict: The aggregated results.
    
    Raises:
        AssertionError: If the keys in the 'results' dictionary are inconsistent across the objects.
    """
    # Assert whether all sublists have the same keys
    first_keys = list(objs[0]['results'].keys())
    assert all([list(obj['results'].keys()) == first_keys for obj in objs])

    gathered_results = {}
    total_n = sum([obj['length'] for obj in objs])
    for key in first_keys:
        gathered_results[key] = sum(obj['results'][key] * obj['length'] for obj in objs) / total_n
    
    return gathered_results



def init_model_config(config) -> dict:
    """
    Make additional kwags for Huggingface model initialization.

    Args:
        config: Configuration object containing training settings.
        load16bit (bool): Whether loading to 16-bit or not.

    Returns:
        dict: Additional kwags
    """
    if config.bit in [4, 8]:
        assert config.peft_config_path, colorstr('red', 'If you quantize the model, you need LoRA etc. due to gradients upating...')
    
    # Basic
    quant_config = init_quant_config(config)
    kwargs = {
        'quantization_config': quant_config,
        'use_cache': False if config.gradient_checkpointing.activate else True
    }

    # Determine attention mechanism
    if config.attn_implementation:
        kwargs['attn_implementation'] = config.attn_implementation
        if config.is_rank_zero:
            LOGGER.info(f"{colorstr(config.attn_implementation)} attention will be used.")
    
    return kwargs



def get_wrap_policy(config):
    """
    Returns a function for wrapping modules based on the specified wrap policy in the config.

    Args:
        config: Configuration object containing FSDP-related hyperparameters.

    Returns:
        Callable: A function that wraps the modules according to the chosen policy.

    Raises:
        NotImplementedError: If the wrap policy specified in the config is unsupported.
    """
    params = config.fsdp_hyperparameters
    wrap_policy = params.wrap_policy
    if wrap_policy.lower() == 'size_based':
        return functools.partial(size_based_auto_wrap_policy, min_num_params=params.size_based.min_num_params)
    elif wrap_policy.lower() == 'transformer_based':
        return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_modules(params)})
    else:
        raise NotImplementedError

    

def custom_wrap_policy(config, model: nn.Module, device: torch.device) -> nn.Module:
    """
    Applies a custom wrapping policy to the model based on the configuration, specifically 
    using Fully Sharded Data Parallel (FSDP) for leaf modules with float32 parameters.

    Args:
        config: Configuration object containing settings for FSDP and CPU offload.
        model (nn.Module): The model to which the custom wrapping will be applied.
        device (torch.device): The device (CPU or GPU) where the model is located.

    Returns:
        nn.Module: The modified model with FSDP applied to the leaf modules.
    
    Raises:
        AttributeError: If the model does not have the expected attributes or structure.
    """
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
                parent, last_name = _get_parent_module(model, name)
                setattr(parent, 
                        last_name, 
                        FSDP(module, 
                             device_id=device, 
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             cpu_offload=CPUOffload(offload_params=True) if config.fsdp_hyperparameters.cpu_offload else None
                        )
                )
                break       # due to leaf modules
    
    if config.is_rank_zero:
        LOGGER.info(colorstr('Custom Wrapping Process is applied because the quantization model is used'))
    
    return model
