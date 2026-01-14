# TODO: jschoi

import gc
import time
import math

import torch
from torch import distributed as dist

from univlt.config import TrainingConfig
from univlt.trainer import BaseTrainer
from univlt.utils import log, colorstr, TQDM
from univlt.utils.common_utils import *
from univlt.utils.training_utils import *
from univlt.utils.peft_utils import merge_unmerged_checkpoints



class AutoregressiveTrainer(BaseTrainer):
    def __init__(
            self, 
            cfg: TrainingConfig,
            mode: str,
            device,
            multi_gpu_train_type=False,
            use_huggingface_trainer=False,
            resume_path=None,
            **kwargs,
        ):
        super().__init__(
            cfg=cfg,
            mode=mode,
            device=device,
            multi_gpu_train_type=multi_gpu_train_type,
            use_huggingface_trainer=use_huggingface_trainer,
            resume_path=resume_path,
            **kwargs,
        )

            
        
    def do_train(self) -> None:
        self.train_time_start = time.time()
        self.train_cur_step = -1
        if not self.is_update_per_epoch:
            self.epochs = math.ceil(self.steps / len(self.dataloaders['train']))
        
        log(f'Using {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers')
        log(f"Logging results to {colorstr('bold', self.save_dir)}")
        log(f'Starting training for {self.epochs} epochs...')

        if self.is_ddp or self.is_fsdp:
            dist.barrier()

        for epoch in range(self.epochs):
            start = time.time()
            self.epoch = epoch

            log('='*200)

            for phase in self.modes:
                log('Phase: {}'.format(phase))

                if phase == 'train':
                    self.epoch_train(phase, epoch)
                    if self.is_ddp or self.is_fsdp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp or self.is_fsdp:
                        dist.barrier()

            # Clear GPU vRAM at end of epoch to prevent OOM errors
            torch.cuda.empty_cache()
            gc.collect()

            # Early Stopping
            if self.stop:
                break  # must break all DDP ranks
            
            log(f"epoch {epoch+1} time: {time.time() - start} s\n\n\n")

        log(f'{epoch - self.start_epoch + 1} epochs completed in {(time.time() - self.train_time_start) / 3600:.3f} hours.')

        # Finally merge unmerged checkpoints at the end of training
        if self.cfg.peft_train is not None and self.cfg.peft_train.adapter_save_type == 'merge':
            if self.is_rank_zero:
                merge_unmerged_checkpoints(self.wdir, self.model_module)
            
            if self.is_ddp or self.is_fsdp:
                dist.barrier()
            

    def epoch_train(self, 
                    phase: str,
                    epoch: int
        ):
        self.model.train()
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)
        validation_step_interval = int(self.cfg.log_cfg.validation_step_interval_prop * nb)
        
        if self.is_ddp or self.is_fsdp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
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
            user_prompt_masking = False if self.cfg.data_cfg.is_multi_turn and self.cfg.data_cfg.user_prompt_masking_start_step > self.train_cur_step else True
            
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.amp
            ):
                if self.gpu_test:
                    batch['src'] = torch.randint(0, self.tokenizer.vocab_size, batch['src'].size(), dtype=torch.long)
                    batch['label'] = torch.randint(0, self.tokenizer.vocab_size, batch['label'].size(), dtype=torch.long)

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch_size = batch['src'].size(0)   # src is always present whether the model is seq2seq or not
                _, loss = self.model(batch, return_loss=True, user_prompt_masking=user_prompt_masking)
                loss = loss / self.cfg.gradient_accumuate_step
            
            # backward and optimizer step
            self.scaler.scale(loss).backward() if self.amp else loss.backward()
            self.optimizer_step(i, is_last_step=i-1 == nb)
            if not self.is_update_per_epoch:
                self.scheduler.step()

            # logging if update criterion is step
            self.training_logger.update(
                phase, 
                epoch+1, 
                self.train_cur_step, 
                batch_size, 
                **{'train_loss': loss.item() * self.cfg.gradient_accumuate_step, 'lr': cur_lr}
            )
            if self.is_rank_zero:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_log = [loss.item() * self.cfg.gradient_accumuate_step]
                msg = tuple([f'{epoch + 1}/{self.epochs}', mem] + loss_log)
                pbar.set_description(('%15s' * 2 + '%15.4g' * len(loss_log)) % msg)
                
            # break if step is over when the update criterion is step
            if not self.is_update_per_epoch and self.train_cur_step == self.steps:
                break

            # validataion
            if self.train_cur_step != 0 and self.train_cur_step % validation_step_interval == 0 and self.cfg.log_cfg.validation_step_interval_prop != 1:
                self.epoch_validate('validation', epoch)
                
                # early stop
                if self.stop:
                    break

                self.model.train()
                if self.is_ddp or self.is_fsdp:
                    dist.barrier()
        
        # upadate logs
        if self.is_rank_zero:
            self.training_logger.update_phase_end(phase, printing=True)
        
        # scheduler step if update criterion is epoch
        if self.is_update_per_epoch:
            self.scheduler.step()

    
    def epoch_validate(self,
                       phase: str,
                       epoch: int,
                       is_training_now=True
        ):
        def _init_headline():
            header = tuple(['Epoch', 'GPU_mem'] + self.loss_names + self.metrics)
            log(('\n' + '%15s' * (2 + len(self.loss_names) + len(self.metrics))) % header)

        def _get_val_pbar(dloader, nb, is_rank_zero):
            if is_rank_zero:
                _init_headline()
                return TQDM(enumerate(dloader), total=nb)
            return enumerate(dloader)

        with torch.no_grad():
            val_loader = self.dataloaders[phase]
            nb = len(val_loader)
            pbar = _get_val_pbar(val_loader, nb, self.is_rank_zero)

            if (self.is_ddp or self.is_fsdp) and (self.cfg.log_cfg.fast_validation_step_interval or self.cfg.log_cfg.fast_validation_n) and is_training_now:
                val_loader.sampler.set_epoch(epoch)

            model = self.ema.ema or self.model if self.ema else self.model
            model.eval()

            # Validation loop
            for i, batch in pbar:
                if self.cfg.log_cfg.fast_validation_step_interval and i % self.cfg.log_cfg.fast_validation_step_interval != 0:
                    continue                    
                
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.amp or self.cfg.half_inference
                ):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    batch_size = batch['src'].size(0)   # src is always present whether the model is seq2seq or not
                    _, loss = self.model(batch, return_loss=True)

                    # Preparing for model evaluation
                    inference_batch_size = min(batch_size, self.cfg.log_cfg.fast_validation_n) if self.cfg.log_cfg.fast_validation_n else batch_size
                    user_prompt = batch['user_prompt'][:inference_batch_size] if 'user_prompt' in batch else batch['src'][:inference_batch_size]
                    response_gt = batch['response'][:inference_batch_size] if 'response' in batch else None
                    response_pred = self.model_module.inference(
                        src=user_prompt,
                        max_length=self.cfg.max_length,
                        num_return_sequences=1,
                        greedy=True,
                        max_time=self.cfg.generation_max_time,
                        synced_gpus=True if self.is_fsdp else None,
                    ) if response_gt and self.do_generate_answer else None

                # Evaluation
                metric_results = self.metric_evaluation(loss, response_pred, response_gt)
                self.training_logger.update(
                    phase, 
                    epoch, 
                    self.train_cur_step if is_training_now else 0, 
                    inference_batch_size, 
                    **{'validation_loss': loss.item()}, 
                    **metric_results
                )

                # Logging
                if self.is_rank_zero:
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    loss_log = [loss.item()]
                    msg = tuple([f'{epoch+1}/{self.epochs}', mem] + loss_log + [metric_results[k] for k in self.metrics])
                    if self.cfg.inference_result_verbose and response_gt != None and self.do_generate_answer:
                        _init_headline()
                    pbar.set_description(('%15s' * 2 + '%15.4g' * (len(loss_log) + len(self.metrics))) % msg)

                    if self.cfg.inference_result_verbose and response_gt != None and self.do_generate_answer:
                        for u, p, g in zip(user_prompt, response_pred, response_gt):
                            print('\n\n' + '-'*200)
                            print(colorstr('\nPrompt    : ') + u)
                            print(colorstr('\nPrediction: ') + p)
                            print(colorstr('\nGT        : ') + g)
                            print('-'*200 + '\n')
            
            # Upadate logs and save model
            self.training_logger.update_phase_end(phase, printing=self.is_rank_zero)

            # Gather and broadcast the results of all ranks. It works only at DDP and FSDP.
            self.collect_all_ranks(nb)

            # Save checkpoint and update early stopper
            if is_training_now:
                self.save_model()
                self.stop = self.early_stopper_step(epoch+1, self.train_cur_step)
