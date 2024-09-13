from sconf import Config
from transformers import BitsAndBytesConfig

import torch

from utils import colorstr



def init_quant_config(config):
    if config.bit in [4, 8]:
        # Sanity check
        assert config.quant_config, colorstr('red','You have to set qunatization config path')
        quant_config = Config(config.quant_config)

        if config.bit == 4:
            assert 'load_in_4bit' in quant_config and quant_config.load_in_4bit, \
                            f'You have to set {colorstr("load_in_4bit")} to {colorstr("True")}'
        else:
            assert 'load_in_8bit' in quant_config and quant_config.load_in_8bit, \
                            f'You have to set {colorstr("load_in_8bit")} to {colorstr("True")}'
        
        kwargs = preprocess({k: v for k, v in quant_config.items() if v})

        return BitsAndBytesConfig(**kwargs)

    return None


def preprocess(kwargs):
    if 'bnb_4bit_compute_dtype' in kwargs:
        tmp_value = kwargs['bnb_4bit_compute_dtype']
        if tmp_value == 'float32':
            kwargs['bnb_4bit_compute_dtype'] = torch.float32
        elif tmp_value == 'float16':
            kwargs['bnb_4bit_compute_dtype'] = torch.float16
        elif tmp_value == 'bfloat16':
            kwargs['bnb_4bit_compute_dtype'] = torch.bfloat16
    
    return kwargs
