from peft import LoraConfig, get_peft_model

from utils import LOGGER, colorstr




def init_lora_config(config):
    return LoraConfig(
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias=config.bias if config.bias != 'none' else 'none',
                task_type=config.task_type,
                target_modules=config.target_modules if config.target_modules else None,
            )



def apply_peft(model, config, peft_type):
    # model.model is huggingface inherited model
    try:
        model.model = get_peft_model(model.model, config)
    except:
        if peft_type == 'lora':
            print(model.model)
            LOGGER.info(f"{colorstr('red', 'Failed to apply PEFT to the model. Please specify the target modules in the lora config according to the above model architectures.')}")
        else:
            pass
        raise AssertionError
    return model



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    LOGGER.info(f"trainable params: {colorstr(trainable_params)} || all params: {colorstr(all_param)} || trainable: {colorstr(100 * trainable_params / all_param)} %")