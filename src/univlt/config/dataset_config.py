from dataclasses import dataclass
from typing import Optional, Union
import os

from univlt.utils import colorstr



@dataclass
class DataConfig:
    # Training dataset
    data_train_type: list[str]
    data_path: list[str]
    template_dir: str

    # Tokenizing method, TODO: Deprecated, change to automatically use official template according to the model.
    add_bos_token_when_response_start: bool = True
    add_eos_token_when_response_end: bool = True
    pad_token_id: Optional[Union[str, int]] = None     # [add, null, int] if null, tokenizer pad_token_id will not be overrided
    bos_token_id: Optional[Union[str, int]] = None     # [add, null, int] if null, tokenizer bos_token_id will not be overrided
    eos_token_id: Optional[Union[str, int]] = None     # [add, null, int] if null, tokenizer eos_token_id will not be overrided
    cls_token_id: Optional[Union[str, int]] = None     # [add, null, int] if null, tokenizer cls_token_id will not be overrided
    sep_token_id: Optional[Union[str, int]] = None     # [add, null, int] if null, tokenizer sep_token_id will not be overrided
    unk_token_id: Optional[Union[str, int]] = None     # [add, null, int] if null, tokenizer unk_token_id will not be overrided

    # Multi-turn data training method
    is_multi_turn: bool = False
    user_prompt_masking_start_step: int = 0  # It activate when you set the is_multi_turn to True. After this steps, user prompts will be masked during computing loss.

    # Verbose option
    data_verbose: bool = True
    

    def __post_init__(self):
        # Dataset condition sanity check
        if not all(_type in ['sft', 'ar'] for _type in self.data_train_type):
            raise ValueError(colorstr("red", f"DataConfig.data_train_type {self.data_train_type} is not supported. Supported list: ['linear', 'cosine']"))
        if not len(self.data_train_type) == len(self.data_path):
            raise AssertionError(colorstr("red", "Lengths of DataConfig.data_train_type and DataConfig.data_path must be the same"))
        if not os.path.exists(self.template_dir):
            raise FileNotFoundError(colorstr("red", f"DataConfig.template_dir `{self.template_dir}` is not found."))
        
        # Tokenizer option sanity check, TODO: Deprecated
        if not (self.pad_token_id in ['add', None] or isinstance(self.pad_token_id, int)):
            raise TypeError(colorstr("red", "DataConfig.pad_token_id must be [`add`, `None`, int]"))
        if not (self.bos_token_id in ['add', None] or isinstance(self.bos_token_id, int)):
            raise TypeError(colorstr("red", "DataConfig.pad_token_id must be [`add`, `None`, int]"))
        if not (self.eos_token_id in ['add', None] or isinstance(self.eos_token_id, int)):
            raise TypeError(colorstr("red", "DataConfig.pad_token_id must be [`add`, `None`, int]"))
        if not (self.cls_token_id in ['add', None] or isinstance(self.cls_token_id, int)):
            raise TypeError(colorstr("red", "DataConfig.pad_token_id must be [`add`, `None`, int]"))
        if not (self.sep_token_id in ['add', None] or isinstance(self.sep_token_id, int)):
            raise TypeError(colorstr("red", "DataConfig.pad_token_id must be [`add`, `None`, int]"))
        if not (self.unk_token_id in ['add', None] or isinstance(self.unk_token_id, int)):
            raise TypeError(colorstr("red", "DataConfig.pad_token_id must be [`add`, `None`, int]"))






        



