import os
import sys
import datetime
from sconf import Config
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from univlt.utils import colorstr
from univlt.utils.training_utils import choose_proper_resume_model
from univlt.trainer import BaseTrainer
from univlt.config import *



def env_setup():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):    
    # Init config
    config = load_config(args.config)

    # Build configurations
    env_cfg = EnvConfig(
        device=config.device,
        seed=config.seed,
        deterministic=config.deterministic,
        workers=config.workers,
        total_cpu_use=config.total_cpu_use
    )
    data_cfg = DataConfig(
        data_train_type=config.data_train_type,
        data_path=config.data_path,
        template_dir=config.template_dir,
        add_bos_token_when_response_start=config.add_bos_token_when_response_start,
        add_eos_token_when_response_end=config.add_eos_token_when_response_end,
        data_verbose=config.data_verbose,
        is_multi_turn=config.is_multi_turn,
        user_prompt_masking_start_step=config.user_prompt_masking_start_step if config.is_multi_turn else 0,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        cls_token_id=config.cls_token_id,
        sep_token_id=config.sep_token_id,
        unk_token_id=config.unk_token_id,
    )
    log_cfg = LoggingConfig(
        common=config.common,
        metrics=config.metrics,
        fast_validation_n=config.fast_validation_n,
        fast_validation_step_interval=config.fast_validation_step_interval,
        validation_step_interval_prop=config.validation_step_interval_prop,
        tensorboard_logging_interval=config.tensorboard_logging_interval,
    )
    fsdp_cfg = FSDPConfig(
        cpu_offload=config.fsdp_hyperparameters.cpu_offload,
        amp_training=config.fsdp_hyperparameters.amp_training,
        wrap_policy=config.fsdp_hyperparameters.wrap_policy,
        size_based=SizeBasedConfig(
            min_num_params=config.fsdp_hyperparameters.size_based.min_num_params
        ) if config.fsdp_hyperparameters.wrap_policy == 'size_based' else None,
        transformer_based=TransformerBasedConfig(
            min_num_params=config.fsdp_hyperparameters.transformer_based.fsdp_layer_cls
        ) if config.fsdp_hyperparameters.wrap_policy == 'transformer_based' else None,
    )
    peft_cfg = PeftConfig(
        peft_config_path=config.peft_config_path if config.peft_config_path else None,
        bit=config.bit if config.quant_config else None,
        quant_config_path=config.quant_config,
        adapter_save_type=config.adapter_save_type,
    )

    cfg = TrainingConfig(
        project=config.project,
        name=config.name,
        model=config.model,
        dtype=config.bit,
        model_cache_dir=config.model_cache_dir,
        attn_implementation=config.attn_implementation if config.attn_implementation else None,
        batch_size=config.batch_size,
        max_length=config.max_length,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        warmup_momentum=config.warmup_momentum,
        optimizer_step_criterion=config.optimizer_step_criterion,
        epochs=config.epochs if config.optimizer_step_criterion == 'epoch' else None,
        warmup_epochs=config.warmup_epochs if config.optimizer_step_criterion == 'epoch' else None,
        steps=config.steps if config.optimizer_step_criterion == 'step' else None,
        warmup_steps=config.warmup_steps if config.optimizer_step_criterion == 'step' else None,
        gradient_accumuate_step=config.gradient_accumuate_step,
        gradient_checkpointing=config.gradient_checkpointing.activate,
        gradient_checkpoint_type=config.gradient_checkpointing.checkpoint_type if config.gradient_checkpointing.activate else None,
        scheduler_type=config.scheduler_type,
        lr0=config.lr0,
        lrf=config.lrf,
        early_stop_criterion=config.early_stop_criterion,
        half_inference=config.half_inference,
        amp_training=config.amp_training,
        ema_updating=config.ema_updating,
        train_verbose=config.train_verbose,
        inference_result_verbose=config.inference_result_verbose,
        generation_max_time=config.generation_max_time,
        del_logits_after_forward=config.del_logits_after_forward,
        data_cfg=data_cfg,
        env_cfg=env_cfg,
        log_cfg=log_cfg,
        peft_train=peft_cfg if config.peft_config_path else None,
        fsdp_train=fsdp_cfg if config.fsdp_train else None
    )
    cfg.yaml_file = args.config
    cfg.training_stage = args.stage
    
    # Init environment
    env_setup()
    
    # Training (cpu/single_gpu or multi_gpu)
    if len(cfg.env_cfg.device) <= 1 or cfg.env_cfg.device == 'cpu':
        single_gpu_train(args, cfg)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.env_cfg.device))
        ngpus_per_node = len(cfg.env_cfg.device)
        torch.multiprocessing.spawn(multi_gpu_train, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, args))

    
def single_gpu_train(args, cfg: TrainingConfig):
    torch.set_num_threads(cfg.env_cfg.total_cpu_use)
    device = torch.device('cpu') if cfg.env_cfg.device == 'cpu' else torch.device(f'cuda:{cfg.env_cfg.device[0]}')
    if device.type == 'cuda':
        torch.cuda.set_device(cfg.env_cfg.device[0])
    trainer = BaseTrainer(
        cfg, 
        args.mode, 
        device, 
        use_huggingface_trainer=args.use_huggingface_trainer,
        resume_path=choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.resume_model_dir else None,
        adapter_path=args.adapter_path if args.adapter_path else None,
        gpu_test=args.gpu_test,
    )

    if args.mode in ['train', 'resume']:
        trainer.do_train()


def multi_gpu_train(gpu, ngpus_per_node, cfg: TrainingConfig, args):
    torch.set_num_threads(cfg.env_cfg.total_cpu_use // ngpus_per_node)

    # Init distribution
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}', 
        world_size=ngpus_per_node,
        rank=gpu, 
        timeout=datetime.timedelta(seconds=args.ddp_timeout)
    )
    torch.cuda.set_device(gpu)
    torch.distributed.barrier()
    trainer = BaseTrainer(
        cfg,
        args.mode,
        gpu,
        multi_gpu_train_type='fsdp' if cfg.fsdp_train is not None else 'ddp',
        resume_path=choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.resume_model_dir else None,
        adapter_path=args.adapter_path if args.adapter_path else None,
        gpu_test=args.gpu_test
    )

    if args.mode in ['train', 'resume']:
        trainer.do_train()





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False)
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'resume'])
    parser.add_argument('-r', '--resume_model_dir', default=None, type=str, required=False)
    parser.add_argument('-a', '--adapter_path', default=None, type=str, required=False)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
    parser.add_argument('-s', '--stage', type=int, default=0, required=False)
    parser.add_argument('-p', '--port', type=str, default='10001', required=False)
    parser.add_argument('--ddp_timeout', type=int, default=86400, required=False)   # 24 hours
    parser.add_argument('--use_huggingface_trainer', action='store_true')
    parser.add_argument('--gpu_test', action='store_true')
    args = parser.parse_args()

    # Sanity checks
    if args.resume_model_dir or args.adapter_path:
        assert args.mode == 'resume', colorstr('red', 'Please set mode to resume..')
    
    if args.mode == 'train':
        assert args.config, colorstr('red', 'config file is required for training..')
        main(args)
    elif args.mode == 'resume':
        assert args.config, colorstr('red', 'config file is required for training..')
        main(args)

    