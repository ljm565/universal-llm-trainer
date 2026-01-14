from typing import Optional
from dataclasses import dataclass, field



@dataclass
class LoggingConfig:
    common: list[str] = field(default_factory=lambda: ['train_loss', 'validation_loss', 'lr'])
    metrics: list[str] = field(default_factory=lambda: ['ppl', 'bleu', 'edit_distance', 'rouge', 'meteor'])
    fast_validation_n: Optional[int] = None                         # [null, 1, 2, ...], Only the number of data of the set value is evaluated per step
    fast_validation_step_interval: int = 5                          # [null, 1, 2, ...], if null, all validation steps will be executed
    validation_step_interval_prop: int = 1                          # setting between 0 and 1 values
    tensorboard_logging_interval: int = 1
