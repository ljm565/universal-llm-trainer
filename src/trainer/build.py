import os
from sconf import Config
from peft import prepare_model_for_kbit_training

import torch
from torch.utils.data import distributed, DataLoader, ConcatDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
)

from data_collection import NMTDataset
from utils import RANK, LOGGER, colorstr
from utils.data_utils import seed_worker, choose_proper_dataset
from utils.peft_utils import init_lora_config, apply_peft, print_trainable_parameters
from utils.filesys_utils import pickle_load
from utils.training_utils import get_wrap_policy, custom_wrap_policy

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def build_nmt_dataset(config, path, tokenizer):
    return NMTDataset(config=config,
                      path=path,
                      tokenizer=tokenizer)



def build_llm_dataset(config, tokenizer, mode):
    dataset_dict = {}
    datasets = [path.split('/')[-1] for path in config.data_path]
    dataset_paths = [os.path.join(p, d + '.pkl') for p, d in zip(config.data_path, datasets)]
    if not config.template_dir:
        template_paths = [os.path.join(p, 'templates') for p in config.data_path]
    else:
        template_paths = [config.template_dir] if isinstance(config.template_dir, str) else config.template_dir
    
    if not all([os.path.exists(p) for p in template_paths]) and config.is_rank_zero:
        raise FileNotFoundError(LOGGER.info(colorstr('red', 'Template directory is not found.')))
    
    dataset_classes = [choose_proper_dataset(d) for d in datasets]

    for i in range(len(datasets)):
        raw_data = pickle_load(dataset_paths[i])
        for state in mode:
            data = [raw_data[name][state] for name in raw_data.keys() if raw_data[name][state] is not None]
            
            # None case
            if len(data) == 0:
                continue
            
            dset = dataset_classes[i](
                mode=state,
                config=config,
                data=sum(data, []),
                tokenizer=tokenizer,
                template_dir=template_paths[i],
                name=datasets[i]
            )

            if state not in dataset_dict:
                dataset_dict[state] = [dset]
            else:
                dataset_dict[state].append(dset)

    # concatenate multiple datasets' class
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



def get_data_loader(config, tokenizer, mode, is_ddp=False):
    if config.train_type == 'nmt':
        paths = {m: os.path.join(config.data_path, m + '.csv') for m in mode}
        datasets = {m: build_nmt_dataset(config,
                                         paths[m], 
                                         tokenizer) for m in mode}
        dataloaders = {m: build_dataloader(datasets[m], 
                                           config.batch_size, 
                                           min([config.workers, config.total_cpu_use]),
                                           shuffle=(m == 'train' or config.fast_validation_n is not None or config.fast_validation_step_interval is not None), 
                                           is_ddp=is_ddp) for m in mode}
    elif config.train_type == 'llm':
        datasets = build_llm_dataset(config, tokenizer, mode)
        dataloaders = {m: build_dataloader(datasets[m], 
                                           config.batch_size, 
                                           min([config.workers, config.total_cpu_use]),
                                           shuffle=(m == 'train' or config.fast_validation_n is not None or config.fast_validation_step_interval is not None), 
                                           is_ddp=is_ddp) for m in mode}
    return dataloaders



def get_model(config, device):
    if config.train_type == 'nmt':
        from models import En2KoNMT
        model = En2KoNMT(config, device)
        tokenizer = model.tokenizer
    elif config.train_type == 'llm':
        if config.model.lower() == 'bagel':
            from models import Bagel
            model = Bagel(config, device)
            tokenizer = model.tokenizer
        elif config.model.lower() == 'kopolyglot':
            from models import KoPolyglot
            model = KoPolyglot(config, device)
            tokenizer = model.tokenizer
        elif config.model.lower() == 't3q_solar':
            from models import T3QSolar
            model = T3QSolar(config, device)
            tokenizer = model.tokenizer
        elif config.model.lower() == 'kogemma':
            from models import KoGemma
            model = KoGemma(config, device)
            tokenizer = model.tokenizer
        elif config.model.lower() == 'gemma':
            from models import Gemma
            model = Gemma(config, device)
            tokenizer = model.tokenizer
        elif config.model.lower() in ['llama3', 'llama3.1']:
            from models import Llama3
            model = Llama3(config, device)
            tokenizer = model.tokenizer
        elif config.model.lower() == 'phi3':
            from models import Phi3
            model = Phi3(config, device)
            tokenizer = model.tokenizer
        else:
            raise NotImplementedError
    else:
        raise AssertionError(f'Invalid train_type: {config.train_type}')
    
    # preparing for bits training
    if model.is4bit or model.is8bit:
        try:
            model = prepare_model_for_kbit_training(model)
        except:
            LOGGER.warning(colorstr('yellow', 'Quantized model preparation is failed. It will not be a problem.'))

    return model, tokenizer



def get_peft_model(model, config):
    peft_config = Config(config.peft_config_path)
    peft_type = peft_config.type

    if peft_type == 'lora':
        lora_config = init_lora_config(peft_config)
        model = apply_peft(model, lora_config, peft_type)
    else:
        raise NotImplementedError
    
    # logs
    if config.is_rank_zero:
        print_trainable_parameters(model)
        LOGGER.info(f'Applied {colorstr(peft_type)} to the model.')
    return model



def get_wrapped_model(config, model, device):
    # Neither quantized nor PEFT case
    if model.is32bit and not config.peft_config_path:
        model = FSDP(model, 
                     auto_wrap_policy=get_wrap_policy(config), 
                     device_id=device, 
                     sharding_strategy=ShardingStrategy.FULL_SHARD,
                     cpu_offload=CPUOffload(offload_params=True) if config.fsdp_hyperparameters.cpu_offload else None
                )
    # Quantized case
    else:
        model = custom_wrap_policy(config, model, device)
    return model