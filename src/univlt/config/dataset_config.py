from dataclasses import dataclass
from typing import Optional, Union
import os

from univlt.utils import colorstr



@dataclass
class DataConfig:
    # Training dataset
    data_train_type: list[str]
    data_path: list[str]
    template_path: str

    # Multi-turn data training method
    is_multi_turn: bool = False
    user_prompt_masking_start_step: int = 0  # It activate when you set the is_multi_turn to True. After this steps, user prompts will be masked during computing loss.

    # Verbose option
    data_verbose: bool = True
    

    def __post_init__(self):
        # Dataset condition sanity check
        if not all(_type in ['qa', 'ar'] for _type in self.data_train_type):
            raise ValueError(colorstr("red", f"DataConfig.data_train_type {self.data_train_type} is not supported. Supported list: ['linear', 'cosine']"))
        if not len(self.data_train_type) == len(self.data_path):
            raise AssertionError(colorstr("red", "Lengths of DataConfig.data_train_type and DataConfig.data_path must be the same"))
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(colorstr("red", f"DataConfig.template_path `{self.template_path}` is not found."))
