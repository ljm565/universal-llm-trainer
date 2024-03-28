import time
import math
import transformers

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
from torch import distributed as dist

from tools import ModelEMA, Evaluator, TrainingLogger
from trainer.build import get_data_loader, get_model, get_peft_model
from utils import (
    OPTIM_CRITERION, 
    OPTIM_CRITERION_MSG,
    SCHEDULER_TYPE,
    SCHEDULER_MSG, 
    RANK, LOGGER,
    colorstr, init_seeds,
    TQDM
)
from utils.training_utils import one_cycle, draw_training_lr_curve, lr_warmup, init_train_progress_bar
from utils.filesys_utils import yaml_save, make_project_dir


__version__ = '0.0.1'



class Trainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            is_ddp=False,
            use_huggingface_trainer=False,
            resume_path=None,
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.is_ddp = is_ddp
        self.is_rank_zero = True if not self.is_ddp or (self.is_ddp and device == 0) else False
        self.config = config
        self.world_size = len(self.config.device) if self.is_ddp else 1
        self.amp = True if self.config.amp_training or self.config.load_unnecessary_half else False
        self.ema = self.config.ema_updating
        self.epochs = self.config.epochs
        self.steps = self.config.steps
        self.optimizer_step_criterion = self.config.optimizer_step_criterion
        self.scheduler_type = self.config.scheduler_type
        self.metrics = self.config.metrics
        self.save_dir = make_project_dir(self.config, self.is_rank_zero)
        self.wdir = self.save_dir / 'weights'
        self.config.is_rank_zero = self.is_rank_zero
        self.loss_names = ['cross_entropy']
        self.train_verbose = self.config.train_verbose
        self.use_huggingface_trainer = use_huggingface_trainer
        self.resume_path = resume_path

        # sanity check
        assert self.optimizer_step_criterion in OPTIM_CRITERION, \
            OPTIM_CRITERION_MSG + f' but got {colorstr(self.optimizer_step_criterion)}'
        assert self.scheduler_type in SCHEDULER_TYPE, \
            SCHEDULER_MSG + f' but got {colorstr(self.scheduler_type)}'
        self.is_update_per_epoch = True if self.optimizer_step_criterion == 'epoch' else False

        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args
        
        # init model, dataset, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['validation']
        self.model, self.tokenizer = self._init_model(self.config, self.mode)
        self.evaluator = Evaluator(self.tokenizer)
        self.training_logger = TrainingLogger(self.config)
        self.dataloaders = get_data_loader(self.config, self.tokenizer, self.modes, self.is_ddp)
        if self.use_huggingface_trainer:
            self.do_train = self.huggingface_trainer
            return 
        
        # init criterion, optimizer, scheduler
        if self.is_training_mode:
            self.lr0 = self.config.lr0
            self.model._init_criterion()
            self.scaler = amp.GradScaler(enabled=self.amp) if self.amp else None
            self.ema = ModelEMA(self.model) if self.ema else None
            self.start_epoch = 0
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr0, betas=(self.config.momentum, 0.999), weight_decay=self.config.weight_decay)
            all_steps_n = self.epochs if self.is_update_per_epoch else self.steps
            self.warmup_steps_n = max(0, self.config.warmup_epochs if self.is_update_per_epoch else self.config.warmup_steps)
            if self.scheduler_type == 'cosine':
                self.lf = one_cycle(1, self.config.lrf, all_steps_n)
            elif self.scheduler_type == 'linear':
                self.lf = lambda x: (1 - (x - self.warmup_steps_n) / (all_steps_n - self.warmup_steps_n)) * (1.0 - self.config.lrf) + self.config.lrf
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
            self.scheduler.last_epoch = self.start_epoch - 1  # do not move
            if self.is_rank_zero and self.train_verbose:
                draw_training_lr_curve(self.config, self.lf, all_steps_n, self.warmup_steps_n, self.is_ddp, self.world_size)
            self.stop = False


    def _init_model(self, config, mode):
        # init model and tokenizer
        model, tokenizer = get_model(config, self.device)

        # init peft
        if config.peft_config_path:
            model = get_peft_model(model, config)
        else:
            LOGGER.info(f'PEFT is not applied.')

        # resume model
        if mode == 'resume':
            LOGGER.info(f'Resumed model: {colorstr(self.resume_path)}')
            checkpoints = torch.load(self.resume_path, map_location=self.device)
            model.load_state_dict(checkpoints['model'])

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
        
        return model, tokenizer
    

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        if self.amp:
            self.scaler.unscale_(self.optimizer)  # unscale gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
            self.optimizer.step()

        self.optimizer.zero_grad()

        if self.ema:
            self.ema.update(self.model)
            
        
    def do_train(self) -> None:
        self.train_time_start = time.time()
        self.train_cur_step = -1
        if not self.is_update_per_epoch:
            self.epochs = math.ceil(self.steps / len(self.dataloaders['train']))
        
        if self.is_rank_zero:
            LOGGER.info(f'Using {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', self.save_dir)}\n"
                        f'Starting training for {self.epochs} epochs...\n')

        if self.is_ddp:
            dist.barrier()

        for epoch in range(self.epochs):
            start = time.time()
            self.epoch = epoch

            if self.is_rank_zero:
                print('-'*100)

            for phase in self.modes:
                if self.is_rank_zero:
                    print('Phase: {}'.format(phase))

                if phase == 'train':
                    self.epoch_train(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()

            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if self.is_ddp:  # if DDP training
                broadcast_list = [self.stop if self.is_rank_zero else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if not self.is_rank_zero:
                    self.stop = broadcast_list[0]
            
            if self.stop:
                break  # must break all DDP ranks
            
            if self.is_rank_zero:
                print(f"\nepoch {epoch+1} time: {time.time() - start} s\n")
                print('\n'*2)

        if RANK in (-1, 0) and self.is_rank_zero:
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            

    def epoch_train(self, 
                    phase: str,
                    epoch: int
        ):
        self.model.train()
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)
        validation_step_interval = int(self.config.validation_step_interval_prop * nb)
        
        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
        if RANK in (-1, 0):
            pbar = init_train_progress_bar(train_loader, self.is_rank_zero, self.loss_names, nb)
    
        # training loop
        self.optimizer.zero_grad()
        for i, batch in pbar:
            # Warmup
            self.train_cur_step += 1
            warmup_step_or_epoch = epoch if self.is_update_per_epoch else self.train_cur_step
            if warmup_step_or_epoch <= self.warmup_steps_n:
                self.optimizer.param_groups[0]['lr'] = lr_warmup(warmup_step_or_epoch, self.warmup_steps_n, self.lr0, self.lf)
            cur_lr = self.optimizer.param_groups[0]['lr']
            
            with torch.cuda.amp.autocast(self.amp):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch_size = batch['src'].size(0)   # src is always present whether the model is seq2seq or not
                _, loss = self.model(batch, return_loss=True)
            
            # backward and optimizer step
            self.scaler.scale(loss).backward() if self.amp else loss.backward()
            self.optimizer_step()
            if not self.is_update_per_epoch:
                self.scheduler.step()

            # logging if update criterion is step
            if RANK in (-1, 0) and self.is_rank_zero:
                self.training_logger.update(phase, epoch+1, self.train_cur_step, batch_size, **{'train_loss': loss.item(), 'lr': cur_lr})
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_log = [loss.item()]
                msg = tuple([f'{epoch + 1}/{self.epochs}', mem] + loss_log)
                pbar.set_description(('%15s' * 2 + '%15.4g' * len(loss_log)) % msg)
                
            # break if step is over when the update criterion is step
            if not self.is_update_per_epoch and self.train_cur_step == self.steps:
                break

            # validataion
            if self.train_cur_step != 0 and self.train_cur_step % validation_step_interval == 0 and self.config.validation_step_interval_prop != 1:
                self.epoch_validate('validation', epoch, middle_validation=True)
                self.model.train()
                if self.is_ddp:
                    dist.barrier()
            if i == 100:
                break
        # upadate logs
        if RANK in (-1, 0) and self.is_rank_zero:
            self.training_logger.update_phase_end(phase, printing=True)
        
        # scheduler step if update criterion is epoch
        if self.is_update_per_epoch:
            self.scheduler.step()

    
    def epoch_validate(self,
                       phase: str,
                       epoch: int,
                       middle_validation=False
        ):
        def _get_val_pbar(dloader, nb, middle_validation):
            # if not middle_validation:
            header = tuple(['Epoch', 'GPU_mem'] + self.loss_names + self.metrics)
            LOGGER.info(('\n' + '%15s' * (2 + len(self.loss_names) + len(self.metrics))) % header)
            return TQDM(enumerate(dloader), total=nb)
            # return enumerate(dloader)

        with torch.no_grad():
            if RANK in (-1, 0) and self.is_rank_zero:
                val_loader = self.dataloaders[phase]
                nb = len(val_loader)
                pbar = _get_val_pbar(val_loader, nb, middle_validation)

                model = self.ema.ema or self.model if self.ema else self.model
                model = model.half() if self.config.half_inference else model.float()
                model.eval()

                # validation loop
                for i, batch in pbar:
                    if self.config.fast_validation_step_interval and i % self.config.fast_validation_step_interval != 0:
                        continue                    

                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    batch_size = batch['src'].size(0)   # src is always present whether the model is seq2seq or not
                    _, loss = self.model(batch, return_loss=True)

                    # preparing for model evaluation
                    inference_batch_size = min(batch_size, self.config.fast_validation_n) if self.config.fast_validation_n else batch_size
                    user_prompt = batch['user_prompt'][:inference_batch_size] if 'user_prompt' in batch else batch['src'][:inference_batch_size]
                    response_gt = batch['response'][:inference_batch_size] if 'response' in batch else None
                    response_pred = self.model.inference(user_prompt, max_length=self.config.max_length, num_return_sequences=1, greedy=True)

                    # evaluation and logging
                    metric_results = self.metric_evaluation(loss, response_pred, response_gt)
                    self.training_logger.update(phase, epoch, self.train_cur_step, inference_batch_size, **{'validation_loss': loss.item()}, **metric_results)

                    # logging
                    # if not middle_validation:
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    loss_log = [loss.item()]
                    msg = tuple([f'{epoch+1}/{self.epochs}', mem] + loss_log + [metric_results[k] for k in self.metrics])
                    pbar.set_description(('%15s' * 2 + '%15.4g' * (len(loss_log) + len(self.metrics))) % msg)

                # upadate logs and save model
                self.training_logger.update_phase_end(phase, printing=True)
                self.training_logger.save_model(self.wdir, self.model)
                self.training_logger.save_logs(self.save_dir)

                            
    def metric_evaluation(self, loss, response_pred, response_gt):
        metric_results = {k: 0 for k in self.metrics}
        for m in self.metrics:
            if m == 'ppl':
                metric_results[m] = self.evaluator.cal_ppl(loss.item())
            elif m == 'bleu':
                metric_results[m] = self.evaluator.cal_bleu_score(response_pred, response_gt)
            elif m == 'rouge':
                metric_results[m] = self.evaluator.cal_rouge_score(response_pred, response_gt, n=None)
            elif m == 'meteor':
                metric_results[m] = self.evaluator.cal_meteor_score(response_pred, response_gt)
            elif m == 'edit_distance':
                metric_results[m] = self.evaluator.cal_edit_distance(response_pred, response_gt)
            else:
                LOGGER.warning(f'{colorstr("red", "Invalid key")}: {m}')
        
        return metric_results
    

    def huggingface_trainer(self):
        import pandas as pd
        from datasets import Dataset, DatasetDict
        from data_collection import huggingface_arc_generator
        from transformers import DataCollatorForLanguageModeling, TrainingArguments
        
        datasets = DatasetDict({k: Dataset.from_pandas(pd.DataFrame(data=v.dataset.data)) for k, v in self.dataloaders.items()})
        templates = self.dataloaders['train'].dataset.templates
        responses = self.dataloaders['train'].dataset.responses
        instructions = self.dataloaders['train'].dataset.instructions

        fn_kwargs={"templates": templates, "responses": responses, "instructions":instructions, "tokenizer": self.tokenizer.tokenizer}
        datasets = {k: v.shuffle().map(huggingface_arc_generator, batched=False, fn_kwargs=fn_kwargs) if k == 'train' else v.map(huggingface_arc_generator, batched=False, fn_kwargs=fn_kwargs) for k, v in datasets.items()}
        
        trainer = transformers.Trainer(
            model=self.model.model,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'] if 'validation' in datasets else None,
            args=TrainingArguments(
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=1,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.steps,
                learning_rate=self.config.lr0,
                fp16=self.amp,
                logging_steps=1,
                output_dir='outputs'
            ),
            data_collator=DataCollatorForLanguageModeling(self.tokenizer.tokenizer, mlm=False)
        )

        trainer.train()