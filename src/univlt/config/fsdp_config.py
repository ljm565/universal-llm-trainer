from typing import Optional
from dataclasses import dataclass, field

from univlt.utils import colorstr



@dataclass
class SizeBasedConfig:
    min_num_params: int = 10000


@dataclass
class TransformerBasedConfig:
    fsdp_layer_cls = None


@dataclass
class FSDPConfig:
    cpu_offload: bool = True
    amp_training: bool = True
    wrap_policy: str = 'size_based'     # [size_based, transformer_based]
    size_based: SizeBasedConfig = field(default_factory=SizeBasedConfig)
    transformer_based: Optional[TransformerBasedConfig] = None


    def __post_init__(self):
        if self.wrap_policy not in ['size_based', 'transformer_based']:
            raise ValueError(colorstr("red", f"FSDPConfig.wrap_policy {self.wrap_policy} is not supported. Supported list: ['size_based', 'transformer_based']"))
        
        # Wrap policy sanity check
        if self.wrap_policy == "size_based":
            if self.size_based is None:
                raise ValueError("size_based config must be provided when wrap_policy='size_based'")
        else:
            if self.transformer_based is None:
                raise ValueError("transformer_based config must be provided when wrap_policy='transformer_based'")
        


