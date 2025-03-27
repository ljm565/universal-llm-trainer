import os
from copy import deepcopy
from peft import LoraConfig, get_peft_model

import torch

from utils import log, colorstr



def init_lora_config(config) -> LoraConfig:
    """
    Initializes the LoRA configuration based on the provided configuration settings.

    Args:
        config (Any): Configuration object that contains LoRA parameters.
            - config.r (int): Rank for the LoRA decomposition.
            - config.lora_alpha (float): Scaling factor for the LoRA weight matrices.
            - config.lora_dropout (float): Dropout rate for LoRA layers.
            - config.bias (str): Bias configuration, can be 'none' or other specific settings.
            - config.task_type (str): The task type for the model.
            - config.target_modules (list, optional): Specific model layers for applying LoRA.

    Returns:
        LoraConfig: A configured LoRA configuration object.
    """
    return LoraConfig(
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias=config.bias if config.bias != 'none' else 'none',
                task_type=config.task_type,
                target_modules=config.target_modules if config.target_modules else None,
            )



def apply_peft(model, config, peft_type: str):
    """
    Applies PEFT (Parameter-Efficient Fine-Tuning) to the model using the specified configuration.

    Args:
        model (torch.nn.Module): The model to which PEFT is applied.
        config (Config): The configuration object for the PEFT method (e.g., LoRAConfig).
        peft_type (str): The type of PEFT to apply, such as 'lora'.

    Returns:
        Any: The model with PEFT applied.

    Raises:
        AssertionError: If applying PEFT fails.

    Notes:
        - `model.model` is Huggingface inherited model
    """
    try:
        model.model = get_peft_model(model.model, config)
    except:
        if peft_type == 'lora':
            log(f'\n{model.model}')
            log('Failed to apply PEFT to the model. Please specify the target modules in the lora config according to the above model architectures.', level='error')
        else:
            # TODO: Add other PEFT types
            pass
        raise AssertionError
    log(f'\n{model}')
    return model



def print_trainable_parameters(model) -> None:
    """
    Prints the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model for which trainable parameters will be calculated.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    log(f"trainable params: {colorstr(trainable_params)} || all params: {colorstr(all_param)} || trainable: {colorstr(100 * trainable_params / all_param)} %")



def merge_unmerged_checkpoints(wdir:str, model:torch.nn.Module) -> None:
    """
    Merge the unmerged checkpoints.

    Args:
        wdir (str): Weight checkpoint directory.
        model (torch.nn.Module): Models for loading saved checkpoints.
    """
    device = torch.device('cpu')
    checkpoint_dirs = [os.path.join(wdir, checkpoint) for checkpoint in os.listdir(wdir)]
    model = model.to(device)
    
    for checkpoint_dir in checkpoint_dirs:
        cloned_model = deepcopy(model)
        checkpoint = torch.load(checkpoint_dir, map_location=device)
        cloned_model.load_state_dict(checkpoint['model'])
        cloned_model.model = cloned_model.model.merge_and_unload()
        checkpoint['model'] = cloned_model.state_dict()
        torch.save(checkpoint, checkpoint_dir)
        log(f'Merged and saved {checkpoint_dir}')
