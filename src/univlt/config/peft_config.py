from dataclasses import dataclass
from typing import Optional, Union

from univlt.utils import colorstr



@dataclass
class PeftConfig:
    # Adapater
    peft_config_path: str
    
    # Quantization
    bit: Optional[Union[str, int]] = None
    quant_config_path: Optional[str] = None
    
    # Saving type
    adapter_save_type: str = 'adapter_only'


    def __post_init__(self):
        # Adapter saving type sanity check
        if self.adapter_save_type not in ['merge', 'adapter_only']:
            raise ValueError(colorstr("red", f"PeftConfig.adapter_save_type {self.adapter_save_type} is not supported. Supported list: ['merge', 'adapter_only']"))
        
        # Quantization sanity check
        if self.bit is not None and self.bit in [4, 8]:
            if self.quant_config_path is None:
                raise ValueError(colorstr("red", "PeftConfig.quant_config must be specified by user"))
        

        


