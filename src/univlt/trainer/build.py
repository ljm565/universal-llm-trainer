import os
from sconf import Config
from datasets import concatenate_datasets
from peft import prepare_model_for_kbit_training

import torch
from torch.utils.data import distributed, DataLoader, ConcatDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
)

from univlt.config import TrainingConfig, PeftConfig
from univlt.utils import RANK, log, colorstr
from univlt.utils.data_utils import seed_worker, choose_proper_dataset
from univlt.utils.peft_utils import init_lora_config, apply_peft, print_trainable_parameters
from univlt.utils.filesys_utils import pickle_load
from univlt.utils.training_utils import get_wrap_policy, custom_wrap_policy

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # Global pin_memory for dataloaders



def build_llm_dataset(cfg: TrainingConfig, tokenizer, mode):
    dataset_dict = {}
    datasets = [path.split('/')[-1] for path in cfg.data_cfg.data_path]
    dataset_paths = [os.path.join(p, d + '.pkl') for p, d in zip(cfg.data_cfg.data_path, datasets)]
    
    dataset_classes = [choose_proper_dataset(d) for d in cfg.data_cfg.data_train_type]

    for i in range(len(datasets)):
        raw_data = pickle_load(dataset_paths[i])
        for state in mode:
            data = [raw_data[name][state] for name in raw_data.keys() if raw_data[name][state] is not None]
            
            # None case
            if len(data) == 0:
                continue
            
            dset = dataset_classes[i](
                mode=state,
                cfg=cfg,
                data=sum(data, []) if isinstance(data[0], list) else concatenate_datasets(data),
                tokenizer=tokenizer,
                template_path=cfg.data_cfg.template_path,
                name=datasets[i]
            )

            if state not in dataset_dict:
                dataset_dict[state] = [dset]
            else:
                dataset_dict[state].append(dset)

    # Concatenate multiple datasets' class
    for state, dsets in dataset_dict.items():
        dataset_dict[state] = ConcatDataset(dsets) if len(dsets) > 1 else dsets[0]

    return dataset_dict



def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    # nd = torch.cuda.device_count()  # number of CUDA devices
    # nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=workers,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)



def get_data_loader(cfg: TrainingConfig, tokenizer, mode, is_ddp=False):
    datasets = build_llm_dataset(cfg, tokenizer, mode)
    dataloaders = {m: build_dataloader(datasets[m], 
                                        cfg.batch_size, 
                                        min([cfg.env_cfg.workers, cfg.env_cfg.total_cpu_use]),
                                        shuffle=(m == 'train' or cfg.log_cfg.fast_validation_n is not None or cfg.log_cfg.fast_validation_step_interval is not None), 
                                        is_ddp=is_ddp) for m in mode}
    return dataloaders



def get_model(cfg: TrainingConfig, device):
    if 'llama-3' in cfg.model.lower():
        from univlt.models import Llama3
        model = Llama3(cfg, device)
        tokenizer = model.tokenizer
    
    elif 'llama-2' in cfg.model.lower():
        from univlt.models import Llama2
        model = Llama2(cfg, device)
        tokenizer = model.tokenizer
    
    elif 'gemma' in cfg.model.lower():
        # Models released after Gemma 2
        if any(model in cfg.model.lower() for model in ['gemma-2', 'gemma-3']):
            from univlt.models import Gemma2
            model = Gemma2(cfg, device)
            tokenizer = model.tokenizer
        # Gemma 1 model
        else:
            from univlt.models import Gemma
            model = Gemma(cfg, device)
            tokenizer = model.tokenizer
    
    elif 'phi-3' in cfg.model.lower():
        from univlt.models import Phi3
        model = Phi3(cfg, device)
        tokenizer = model.tokenizer
    
    elif 'qwen3' in cfg.model.lower():
        from univlt.models import Qwen3
        model = Qwen3(cfg, device)
        tokenizer = model.tokenizer
    
    else:
        raise NotImplementedError
    
    # Preparing for bits training
    if model.bit in [4, 8]:
        try:
            model = prepare_model_for_kbit_training(model)
        except:
            log('Quantized model preparation is failed. It will not be a problem.', level='warning')

    return model, tokenizer



def get_peft_model(model, cfg: PeftConfig):
    peft_config = Config(cfg.peft_config_path)
    peft_type = peft_config.type

    if peft_type == 'lora':
        lora_config = init_lora_config(peft_config)
        model = apply_peft(model, lora_config, peft_type)
    else:
        raise NotImplementedError
    
    # Logging
    print_trainable_parameters(model)
    log(f'Applied {colorstr(peft_type)} to the model.')
    return model



def get_wrapped_model(cfg: TrainingConfig, model, device):
    if not (cfg.peft_train and cfg.peft_train.quant_config_path):
        model = FSDP(model, 
                     auto_wrap_policy=get_wrap_policy(cfg), 
                     device_id=device, 
                     sharding_strategy=ShardingStrategy.FULL_SHARD,
                     cpu_offload=CPUOffload(offload_params=True) if cfg.fsdp_train.cpu_offload else None,
                     mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True) if cfg.fsdp_train.amp_training else None,
                )
    # Quantized case
    else:
        model = custom_wrap_policy(cfg, model, device)
    return model