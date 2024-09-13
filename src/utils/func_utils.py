from sconf import Config

from utils import (
    OPTIM_CRITERION, 
    OPTIM_CRITERION_MSG,
    SCHEDULER_TYPE,
    SCHEDULER_MSG, 
    FSDP_WRAP_TYPE,
    FSDP_WRAP_MSG,
    colorstr
)



def sanity_check(clazz):
    assert clazz.optimizer_step_criterion in OPTIM_CRITERION, \
        OPTIM_CRITERION_MSG + f' but got {colorstr(clazz.optimizer_step_criterion)}'
    assert clazz.scheduler_type in SCHEDULER_TYPE, \
        SCHEDULER_MSG + f' but got {colorstr(clazz.scheduler_type)}'
    if clazz.config.fsdp_train:
        assert clazz.config.fsdp_hyperparameters.wrap_policy in FSDP_WRAP_TYPE, \
            FSDP_WRAP_MSG + f' but got {colorstr(clazz.config.fsdp_hyperparameters.wrap_policy)}'
        
    
def select_training_type(train_type):
    """
    Returns:
        (bool): DDP training
        (bool): FSDP training
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


def replace_none_value(config):
    if isinstance(config, Config):
        return {k: replace_none_value(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_none_value(item) for item in config]
    elif config == 'None':
        return None
    else:
        return config
