from dataclasses import dataclass

from univlt.utils import colorstr



@dataclass
class EnvConfig:
    device: list[int]
    seed: int = 0
    deterministic: bool = True
    workers: int = 1
    total_cpu_use: int = 32


    def __post_init__(self):
        # Device sanity check
        if not isinstance(self.device, list):
            raise TypeError(colorstr("red", "EnvConfig.device must be list[int]"))
        if not all(isinstance(x, int) for x in self.device):
            raise TypeError(colorstr("red", "device must be list[int]"))


