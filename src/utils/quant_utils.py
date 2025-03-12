from sconf import Config
from typing import Optional
from transformers import BitsAndBytesConfig

import torch

from utils import colorstr



def init_quant_config(config: Config) -> Optional[BitsAndBytesConfig]:
    """
    Initializes the quantization configuration for 4-bit or 8-bit quantization.

    Args:
        config (Config): A configuration object containing quantization settings.
            - config.bit (int): The desired quantization bit-width (4 or 8).
            - config.quant_config (str): Path to the quantization configuration file.

    Returns:
        Optional[BitsAndBytesConfig]: The initialized quantization configuration for the model.
            Returns `None` if the quantization bit-width is not 4 or 8.

    Raises:
        AssertionError: If the required quantization configuration path is not provided.
        AssertionError: If `load_in_4bit` or `load_in_8bit` is not properly set in the configuration.
    """
    if config.bit in [4, 8]:
        # Sanity check
        assert config.quant_config, colorstr('red','You have to set qunatization config path')
        quant_config = Config(config.quant_config)

        if config.bit == 4:
            assert 'load_in_4bit' in quant_config and quant_config.load_in_4bit, \
                            f'You have to set {colorstr("red", "load_in_4bit")} to {colorstr("red", "True")}'
        else:
            assert 'load_in_8bit' in quant_config and quant_config.load_in_8bit, \
                            f'You have to set {colorstr("red", "load_in_8bit")} to {colorstr("red", "True")}'
        
        kwargs = preprocess({k: v for k, v in quant_config.items() if v})

        return BitsAndBytesConfig(**kwargs)

    return None



def preprocess(kwargs: dict) -> dict:
    """
    Preprocesses quantization parameters for compatibility with PyTorch and BitsAndBytesConfig.
        - Converts the value of `bnb_4bit_compute_dtype` to the corresponding PyTorch data type.

    Args:
        kwargs (dict): A dictionary of quantization parameters.

    Returns:
        dict: The updated dictionary with properly formatted quantization parameters.
    """
    if 'bnb_4bit_compute_dtype' in kwargs:
        tmp_value = kwargs['bnb_4bit_compute_dtype']
        if tmp_value == 'float32':
            kwargs['bnb_4bit_compute_dtype'] = torch.float32
        elif tmp_value == 'float16':
            kwargs['bnb_4bit_compute_dtype'] = torch.float16
        elif tmp_value == 'bfloat16':
            kwargs['bnb_4bit_compute_dtype'] = torch.bfloat16
    
    return kwargs
