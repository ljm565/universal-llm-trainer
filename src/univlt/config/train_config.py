from typing import Optional, Union
from dataclasses import dataclass, field

from univlt.config import *
from univlt.utils import colorstr



@dataclass
class TrainingConfig:
    # Data configuration
    data_cfg: DataConfig

    # Project directory
    project: str
    name: str
    
    # Model condition
    model: str
    dtype: Union[str, int]
    
    # Hyperparameters
    batch_size: int
    max_length: int
    momentum: float = 0.9
    weight_decay: float = 0.0
    warmup_momentum: float = 0.8

    # Training strategy
    optimizer_step_criterion: str = 'step'
    epochs: Optional[int] = None
    warmup_epochs: Optional[int] = None
    steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    gradient_accumuate_step: int = 32
    gradient_checkpointing: bool = True
    gradient_checkpoint_type: str = 'hf_checkpoint'
    amp_training: bool = False
    ema_updating: bool = False
    
    # Learning rate and scheduler
    scheduler_type: str = 'cosine'
    lr0: float = 5e-5
    lrf: float = 0.01   # Last lr = lr0 * lrf, TODO: lr_p Change name to represent proportion meaning instead of final.

    # Other parameters
    model_cache_dir: Optional[str] = None
    attn_implementation: Optional[str] = None
    early_stop_criterion: int = 5
    half_inference: bool = False
    train_verbose: bool = True
    inference_result_verbose: bool = True
    generation_max_time: int = 200        # Maximum time (seconds) of model's generation function. If null, no time limits. 
    del_logits_after_forward: bool = True  # If True, you can save GPU memroy consumption.

    # Other configurations
    env_cfg: EnvConfig = field(default_factory=EnvConfig)
    log_cfg: LoggingConfig = field(default_factory=LoggingConfig)
    peft_train: Optional[PeftConfig] = None
    fsdp_train: Optional[FSDPConfig] = None


    def __post_init__(self):
        # Scheduler type sanity check
        if self.scheduler_type not in ['cosine', 'linear']:
            raise ValueError(colorstr("red", f"TrainingConfig.scheduler_type {self.scheduler_type} is not supported. Supported list: ['linear', 'cosine']"))
        
        # Sanity check for an optimizer step criteria
        if self.optimizer_step_criterion not in ['step', 'epoch']:
            raise ValueError(colorstr("red", f"TrainingConfig.optimizer_step_criterion {self.optimizer_step_criterion} is not supported. Supported list: ['step', 'epoch']"))
        if self.optimizer_step_criterion == 'step':
            if self.steps is None or self.warmup_steps is None:
                raise ValueError(colorstr("red", "`steps`, `warmup_steps` must be specified by user"))
        elif self.optimizer_step_criterion == 'epoch':
            if self.epochs is None or self.warmup_epochs is None:
                raise ValueError(colorstr("red", "`epochs`, `warmup_epochs` must be specified by user"))
            
        # Gradient checkpointing sanity check
        if self.gradient_checkpointing:
            if self.gradient_checkpoint_type not in ['hf_checkpoint', 'torch_checkpoint']:
                raise ValueError(colorstr("red", f"TrainingConfig.checkpoint_type {self.gradient_checkpoint_type} is not supported. Supported list: ['hf_checkpoint', 'torch_checkpoint']"))

        
        

        
        




    

