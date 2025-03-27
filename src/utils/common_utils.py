from sconf import Config
from typing import Any, Dict, List, Tuple, Union

import torch

from utils import (
    OPTIM_CRITERION, 
    OPTIM_CRITERION_MSG,
    SCHEDULER_TYPE,
    SCHEDULER_MSG, 
    FSDP_WRAP_TYPE,
    FSDP_WRAP_MSG,
    ADAPTER_SAVE_TYPE_MSG,
    ADAPTER_SAVE_TYPE,
    colorstr
)



def sanity_check(clazz: Any) -> None:
    """
    Performs a series of sanity checks on the configuration of the given class.
        - Validates `optimizer_step_criterion` against predefined values.
        - Validates `scheduler_type` against predefined values.
        - Checks `fsdp_train` and `fsdp_hyperparameters.wrap_policy` for FSDP configurations.
        - Ensures `amp_training` is set when `attn_implementation` is used.

    Args:
        clazz (Any): The class instance containing configuration options for training.

    Returns:
        None: Raises `AssertionError` if any of the sanity checks fail.
    """
    assert clazz.optimizer_step_criterion in OPTIM_CRITERION, \
        OPTIM_CRITERION_MSG + f' but got {colorstr(clazz.optimizer_step_criterion)}.'
    assert clazz.scheduler_type in SCHEDULER_TYPE, \
        SCHEDULER_MSG + f' but got {colorstr(clazz.scheduler_type)}.'
    if clazz.config.fsdp_train:
        assert clazz.config.fsdp_hyperparameters.wrap_policy in FSDP_WRAP_TYPE, \
            FSDP_WRAP_MSG + f' but got {colorstr(clazz.config.fsdp_hyperparameters.wrap_policy)}.'
    if clazz.config.attn_implementation:
        assert clazz.config.amp_training or (clazz.config.fsdp_train and clazz.config.fsdp_hyperparameters.amp_training), \
            colorstr('You must set amp_training option to True if you use attn_implementation option.')
    if clazz.config.peft_config_path:
        assert clazz.config.adapter_save_type in ADAPTER_SAVE_TYPE, \
            ADAPTER_SAVE_TYPE_MSG + f' but got {colorstr(clazz.config.adapter_save_type)}.'
        

    
def select_training_type(train_type: Union[bool, str]) -> Tuple[bool, bool]:
    """
    Determines the type of training (DDP or FSDP) based on the provided input.

    Args:
        train_type (Union[bool, str]): A string indicating the type of training ('ddp', 'fsdp', or False).

    Returns:
        Tuple[bool, bool]: A tuple indicating whether DDP training (first element) or FSDP training (second element).

    Raises:
        NotImplementedError: If an unsupported `train_type` is provided.
    """
    if not train_type:
        return False, False
    elif train_type.lower() == 'ddp':
        return True, False
    elif train_type.lower() == 'fsdp':
        return False, True
    else:
        raise NotImplementedError



def wrap_modules(params):
    # TODO: transformer FSDP wrapping function 
    pass    



def replace_none_value(config: Config) -> Union[Dict[str, Any], List[Any], Any, None]:
    """
    Recursively replaces string 'None' values with actual `None` in a configuration object.

    Args:
        config (Union[Config, List[Any], Any]): The configuration object, list, or individual value to be processed.
    
    Returns:
        Union[Dict[str, Any], List[Any], Any, None]: The updated configuration object with 'None' string values replaced by actual `None`.
    """
    if isinstance(config, Config):
        return {k: replace_none_value(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_none_value(item) for item in config]
    elif config == 'None':
        return None
    else:
        return config



def instantiate(parent: Any, string: str) -> Any:
    """
    Converts a string representing a PyTorch dtype into the corresponding PyTorch dtype object.

    Args:
        parent (Any): Paraent library.
        string (str): A string representing a PyTorch dtype (e.g., "float32", "bfloat16", "float16").

    Returns:
        Any: All types of instance

    Raises:
        ValueError: If the input string does not correspond to a valid PyTorch dtype.

    Example:
        >>> dtype = instantiate(torch, "bfloat16")
        >>> print(dtype)
        torch.bfloat16

        >>> dtype = instantiate(torch, "invalid_dtype")
        ValueError: 'invalid_dtype' is not a valid instance.
    """
    if hasattr(parent, string):
        return getattr(parent, string)
    else:
        raise ValueError(colorstr(f"'{string}' is not a valid instance.", "red"))