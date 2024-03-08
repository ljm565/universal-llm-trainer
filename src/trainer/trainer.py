import os
import time
import math
import warnings
import numpy as np
import transformers
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
from torch import distributed as dist

from tools import ModelEMA
from trainer.build import get_data_loader, get_model, get_peft_model
from utils import (
    OPTIM_CRITERION, 
    OPTIM_CRITERION_MSG,
    SCHEDULER_TYPE,
    SCHEDULER_MSG, 
    RANK, LOGGER, LOG_DATA,
    colorstr, init_seeds,
    TQDM
)
from utils.training_utils import one_cycle, draw_training_lr_curve, lr_warmup, init_train_progress_bar, save_model
from utils.filesys_utils import yaml_save, make_project_dir


__version__ = '0.0.1'



class Trainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            is_ddp=False,
            use_huggingface_trainer=False
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = mode
        self.device = torch.device(device)
        self.is_ddp = is_ddp
        self.is_rank_zero = True if not self.is_ddp or (self.is_ddp and device == 0) else False
        self.config = config
        self.amp = True if self.config.amp_training else False
        self.ema = self.config.ema_updating
        self.epochs = self.config.epochs
        self.steps = self.config.steps
        self.optimizer_step_criterion = self.config.optimizer_step_criterion
        self.scheduler_type = self.config.scheduler_type
        self.batch_size = self.config.batch_size
        self.save_dir = make_project_dir(self.config, self.is_rank_zero)
        self.wdir = self.save_dir / 'weights'
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'
        self.loss_names = ['cross_entropy']
        self.train_verbose = self.config.train_verbose
        self.use_huggingface_trainer = use_huggingface_trainer

        # sanity check
        assert self.optimizer_step_criterion in OPTIM_CRITERION, \
            OPTIM_CRITERION_MSG + f' but got {colorstr(self.optimizer_step_criterion)}'
        assert self.scheduler_type in SCHEDULER_TYPE, \
            SCHEDULER_MSG + f' but got {colorstr(self.scheduler_type)}'
        assert all([d in LOG_DATA for d in self.config.log_data]), \
            SCHEDULER_MSG + f' but got {colorstr(self.scheduler_type)}'
        self.is_update_per_epoch = True if self.optimizer_step_criterion == 'epoch' else False
        self.training_log_data = {d: [] for d in self.config.log_data}
        if self.is_update_per_epoch:
            self.training_log_data['epoch'] = []
        else:
            self.training_log_data['step'] = []

        # save the yaml config
        if self.is_rank_zero and self.mode == 'train':
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args
        
        # init model, dataset, dataloader
        self.modes = ['train', 'validation'] if self.mode == 'train' else ['validation']
        self.model, self.tokenizer = self._init_model(self.config, self.mode)
        self.dataloaders = get_data_loader(self.config, self.tokenizer, self.modes, self.is_ddp)
        if self.use_huggingface_trainer:
            self.do_train = self.huggingface_trainer
            return 
        
        # init criterion, optimizer, scheduler
        if self.mode == 'train':
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
            if self.train_verbose:
                draw_training_lr_curve(self.config, self.lf, all_steps_n, self.warmup_steps_n)
            print()


    def _init_model(self, config, mode):
        # init model and tokenizer
        model, tokenizer = get_model(config, self.device)

        # init peft
        if config.peft_config_path:
            model = get_peft_model(model, config)
        else:
            LOGGER.info(f'PEFT is not applied.')


        # model = model.to(self.device).half() if self.half else model.to(self.device).float()

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
        
        # if not mode == 'train':
        #     model.eval()
        # TODO: fusion logic

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
        self.nb = len(self.dataloaders['train'])  # number of batches
        self.world_size = len(self.config.device) if self.is_ddp else 1
        if not self.is_update_per_epoch:
            self.epochs = math.ceil(self.steps / self.nb / self.world_size)  # adjust number of epochs 
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
                    
                    save_model(str(self.save_dir / 'model.pt'), self.model.state_dict())
                
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
                print(f"epoch {epoch+1} time: {time.time() - start} s\n")
                print('\n'*2)

        if RANK in (-1, 0) and self.is_rank_zero:
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            
            self.final_eval()
            if self.trainer_plots:
                plot_results(file=self.csv, function_num=self.config.function_num, on_plot=self.on_plot)  # save results.png
        
        torch.cuda.empty_cache()


    def epoch_train(self, 
                    phase: str,
                    epoch: int
        ):

        self.model.train()
        epoch_loss = 0
        train_loader = self.dataloaders[phase]
        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
        if RANK in (-1, 0):
            pbar = init_train_progress_bar(train_loader, self.is_rank_zero, self.loss_names, self.nb)
    
        # training loop
        self.optimizer.zero_grad()
        for i, batch in pbar:
            # Warmup
            cur_step = epoch if self.is_update_per_epoch else i + self.nb * epoch
            if cur_step <= self.warmup_steps_n:
                self.optimizer.param_groups[0]['lr'] = lr_warmup(cur_step, self.warmup_steps_n, self.lr0, self.lf)
            
            with torch.cuda.amp.autocast(self.amp):
                src_tok, src_mask = batch['src'].to(self.device), batch['src_attention_mask'].to(self.device)
                trg_tok = batch['trg'].to(self.device) if 'trg' in batch else None
                trg_mask = batch['trg_attention_mask'].to(self.device) if 'trg_attention_mask' in batch else None
                label = batch['label'].to(self.device) if 'label' in batch else None
                    
                batch_size = src_tok.size(0)
                _, loss = self.model(
                    (src_tok, src_mask), 
                    (trg_tok, trg_mask),
                    label,
                    return_loss=True
                )
            
            # backward and optimizer step
            self.scaler.scale(loss).backward() if self.amp else loss.backward()
            self.optimizer_step()
            if not self.is_update_per_epoch:
                self.scheduler.step()

            # logging if update criterion is step
            if RANK in (-1, 0) and self.is_rank_zero:
                epoch_loss += loss.item() * batch_size
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_log = [loss.item()]
                msg = tuple([f'{epoch + 1}/{self.epochs}', mem] + loss_log)
                pbar.set_description(('%15s' * 2 + '%15.4g' * len(loss_log)) % msg)
                if not self.is_update_per_epoch:
                    self.logging(cur_step, loss.item())
                
            # break if step is over when the update criterion is step
            if not self.is_update_per_epoch:
                if cur_step == self.steps:
                    break

        # logging if update criterion is epoch
        epoch_loss = epoch_loss / len(train_loader.dataset)
        self.logging(epoch, epoch_loss)
        
        # scheduler step if update criterion is epoch
        if self.is_update_per_epoch:
            self.scheduler.step()

    
    def epoch_validate(self,
                       phase: str,
                       epoch: int
        ):
        with torch.no_grad():
            if RANK in (-1, 0) and self.is_rank_zero:
                val_loader = self.dataloaders[phase]
                pbar = init_train_progress_bar(val_loader, self.is_rank_zero, self.loss_names, self.nb)
                epoch_loss = 0

                # is ema attributes updating really necessary?
                if self.ema:
                    self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                                
                model = self.ema.ema or self.model if self.ema else self.model
                model = model.half() if self.config.half_inference else model.float()
                model.eval()

                # validation loop
                for i, batch in enumerate(pbar):
                    cur_step = epoch if self.is_update_per_epoch else i + self.nb * epoch

                    src_tok, src_mask = batch['src'].to(self.device), batch['src_attention_mask'].to(self.device)
                    trg_tok = batch['trg'].to(self.device) if 'trg' in batch else None
                    trg_mask = batch['trg_attention_mask'].to(self.device) if 'trg_attention_mask' in batch else None
                    label = batch['label'].to(self.device) if 'label' in batch else None
                        
                    batch_size = src_tok.size(0)
                    _, loss = self.model(
                        (src_tok, src_mask), 
                        (trg_tok, trg_mask),
                        label,
                        return_loss=True
                    )

                    # logging
                    epoch_loss += loss.item() * batch_size    
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    loss_log = [loss.item()]
                    msg = tuple([f'{epoch + 1}/{self.epochs}', mem] + loss_log)
                    pbar.set_description(('%15s' * 2 + '%15.4g' * len(loss_log)) % msg)
                    if not self.is_update_per_epoch:
                        self.logging(cur_step, loss.item(), 'validation')

                    # inference
                    if not query_end_loc == None:
                        for j, loc in enumerate(query_end_loc.tolist()):
                            gt_seq = self.tokenizer.decode(src_tok[j][loc:].tolist())
                            pred_seq = self.tokenizer.decode(self.model.generate(src_tok[j][:loc].unsqueeze(0), max_length=self.config.max_length, greedy=True)[0].tolist())
                            print(f'GT: {gt_seq.replace(self.tokenizer.pad_token, "")}')
                            print('-'*100)
                            print(f'Pred: {pred_seq}')
                            print('/n'*2)



                    

                # stats = [self.det_metrics[i].finalize_valiation(i, dt, val_loader) for i in range(len(preds))]
                # model.float()

                # self.fitness = 0
                # all_metrics = {}
                # for i in range(len(preds)):
                #     self.fitness += stats[i].pop('fitness', -self.loss[i].detach().cpu().numpy())  # use loss as fitness measure if not found
                #     results = {k+f'({i})': v for k, v in stats[i].items()}
                #     results.update({**label_loss_items(self.loss_names[i*3:(i+1)*3], self.loss[i].cpu() / len(val_loader), prefix='val')})
                #     metrics = {k: round(float(v), 5) for k, v in results.items()}
                #     all_metrics.update(metrics)
                    
                # if not self.best_fitness or self.best_fitness < self.fitness:
                #     self.best_fitness = self.fitness

                # self.save_metrics(
                #     metrics={
                #         **label_loss_items(
                #             self.loss_names,
                #             torch.cat([loss if loss != None else torch.zeros(self.loss_len[i]).to(self.device) for i, loss in enumerate(self.tlosses)])
                #         ), 
                #         **all_metrics,
                #         **self.lr}
                # )
                # self.stop = self.stopper(epoch + 1, self.fitness)

                # # Save model
                # if self.config.save or (epoch + 1 == self.epochs):
                #     self.save_model()
            

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch + 1] + vals)).rstrip(',') + '\n')


    # def save_model(self):
    #     """Save model training checkpoints with additional metadata."""
    #     import pandas as pd  # scope for faster startup
    #     metrics = {**self.metrics, **{'fitness': self.fitness}}
    #     results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}
    #     ckpt = {
    #         'epoch': self.epoch,
    #         'best_fitness': self.best_fitness,
    #         'model': deepcopy(de_parallel(self.model)).half(),
    #         'ema': deepcopy(self.ema.ema).half(),
    #         'updates': self.ema.updates,
    #         'optimizer': self.optimizer.state_dict(),
    #         'train_args': self.config,  # save as dict
    #         'train_metrics': metrics,
    #         'train_results': results,
    #         'date': datetime.now().isoformat(),
    #         'version': __version__}

        # Save last and best
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')


    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers

                # if f is self.best:
                #     LOGGER.info(f'\nValidating {f}...')
                #     self.validator.args.plots = self.args.plots
                #     self.metrics = self.validator(model=f)
                #     self.metrics.pop('fitness', None)
                #     self.run_callbacks('on_fit_epoch_end')
    

    def logging(self, cur_step, loss, state='train'):
        self.training_log_data['lr'].append(self.optimizer.param_groups[0]['lr'])
        self.training_log_data[f'{state}_loss'].append(loss)
        self.training_log_data['step'].append(cur_step)


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