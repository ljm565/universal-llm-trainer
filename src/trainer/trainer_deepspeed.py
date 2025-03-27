import gc
import time
import math
try:
    import deepspeed
except:
    pass

import torch
from torch.cuda import amp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tools import Evaluator, TrainingLogger, EarlyStopper
from trainer.build import get_data_loader, get_model, get_peft_model, get_wrapped_model
from utils import (
    RANK, LOGGER,
    colorstr, init_seeds,
    TQDM
)
from utils.common_utils import *
from utils.training_utils import *
from utils.filesys_utils import yaml_save, make_project_dir, json_load, json_save


__version__ = '0.0.1'



class TrainerDeepSpeed:
    def __init__(
            self, 
            config,
            args,
            device,
            multi_gpu_train_type=False,
            resume_path=None,
        ):
        self.is_ddp, self.is_fsdp = select_training_type(multi_gpu_train_type)
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = args.mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.is_rank_zero = True #if not multi_gpu_train_type or (multi_gpu_train_type and device == 0) else False
        self.config = config
        self.world_size = len(self.config.device) if multi_gpu_train_type else 1
        self.steps = self.config.steps
        self.optimizer_step_criterion = 'step'
        self.metrics = self.config.metrics
        if self.is_training_mode:
            self.save_dir = make_project_dir(self.config, self.is_rank_zero)
            self.wdir = self.save_dir / 'weights'
        else:
            config.is_training_mode = False
            config.data_verbose = False
            if self.config.fast_validation_n == 'None': 
                self.config.fast_validation_n = None
            if self.config.fast_validation_step_interval == 'None': 
                self.config.fast_validation_step_interval = None
        self.config.is_rank_zero = self.is_rank_zero
        self.loss_names = ['cross_entropy']
        self.train_verbose = self.config.train_verbose
        self.resume_path = resume_path
        self.deepspeed_config = json_load(args.deepspeed_config)
        # self.config.batch_size = self.deepspeed_config['train_batch_size']

        # sanity check
        if 'fp16' in self.deepspeed_config:    
            self.amp = self.deepspeed_config['fp16']['enabled']
        
        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args
            json_save(self.save_dir / 'deepspeed_args.json', self.deepspeed_config)  # save deepspeed config
        
        # init model, dataset, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['validation']
        self.model, self.tokenizer = self._init_model(self.config, self.mode)
        self.evaluator = Evaluator(self.tokenizer)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)
        self.stopper, self.stop = EarlyStopper(self.config.early_stop_criterion), False
        self.dataloaders = get_data_loader(self.config, self.tokenizer, self.modes, self.is_ddp)
        # print(torch.cuda.current_device())
        # sfsd
        
        # init criterion and deepspeed's optimizer and model engine
        if self.is_training_mode:
            self.scaler = amp.GradScaler(enabled=self.amp) if self.amp else None
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                args=args, 
                model=self.model, 
                model_parameters=self.model.parameters(),
            )


    def _init_model(self, config, mode):
        def _resume_model(resume_path, device, is_rank_zero):
            checkpoints = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoints['model'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return model

        # init model, loss function, and tokenizer
        resume_success = False
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        model, tokenizer = get_model(config, self.device)
        model.init_criterion()

        # init peft
        if config.peft_config_path:
            if config.training_stage != 0:
                if config.is_rank_zero:
                    LOGGER.info(f'PEFT is not applied due to training stage.')
            else:
                # resume before applying peft
                if do_resume:
                    try:
                        model = _resume_model(self.resume_path, self.device, config.is_rank_zero)
                        resume_success = True
                    except:
                        pass
                model = get_peft_model(model, config)
        else:
            if config.is_rank_zero:
                LOGGER.info(f'PEFT is not applied.')

        # resume model or resume model after applying peft
        if do_resume and not resume_success:
            model = _resume_model(self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            model = DDP(model, device_ids=[self.device])
        elif self.is_fsdp:
            model = get_wrapped_model(config, model, self.device)

        return model, tokenizer   
    

    def _freeze_model(self):
        if self.config.training_stage:
            # Firstly, changing model dtype
            if self.model_engine.load16bit:
                self.model_engine.half()
            else:
                self.model_engine.float()
            
            # Secondly, freezing layer except for that need to be trained 
            self.model_engine.freeze_layers(self.config.training_stage)

            # Lastly, changing layers dtype those have to be float32
            if self.model_engine.load16bit:
                self.model_engine.mapping_neccessary_32bit()
      
        
    def do_train(self) -> None:
        self.train_time_start = time.time()
        self.train_cur_step = -1
        if not self.is_update_per_epoch:
            self.epochs = math.ceil(self.steps / len(self.dataloaders['train']))
        
        if self.is_rank_zero:
            LOGGER.info(f'Using {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', self.save_dir)}\n"
                        f'Starting training for {self.epochs} epochs...\n')

        if self.is_ddp or self.is_fsdp:
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
                    if self.is_ddp or self.is_fsdp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp or self.is_fsdp:
                        dist.barrier()

            # clears GPU vRAM at end of epoch, can help with out of memory errors
            torch.cuda.empty_cache()
            gc.collect()

            # Early Stopping
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
        self.model_engine.train()
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
            self.train_cur_step += 1
            cur_lr = self.optimizer.param_groups[0]['lr']
            
            with torch.cuda.amp.autocast(self.amp):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch_size = batch['src'].size(0)   # src is always present whether the model is seq2seq or not
                _, loss = self.model_engine(batch, return_loss=True)
                self.model_engine.backward(loss)

            self.model_engine.step()

            # logging if update criterion is step
            self.training_logger.update(
                phase, 
                epoch+1, 
                self.train_cur_step, 
                batch_size, 
                **{'train_loss': loss.item() * self.config.gradient_accumuate_step, 'lr': cur_lr}
            )
            if RANK in (-1, 0) and self.is_rank_zero:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_log = [loss.item()]
                msg = tuple([f'{epoch + 1}/{self.epochs}', mem] + loss_log)
                pbar.set_description(('%15s' * 2 + '%15.4g' * len(loss_log)) % msg)
                
            # break if step is over when the update criterion is step
            if not self.is_update_per_epoch and self.train_cur_step == self.steps:
                break

            # validataion
            if self.train_cur_step != 0 and self.train_cur_step % validation_step_interval == 0 and self.config.validation_step_interval_prop != 1:
                self.epoch_validate('validation', epoch)
                
                # early stop
                if self.stop:
                    break

                self.model.train()
                if self.is_ddp or self.is_fsdp:
                    dist.barrier()
        
        # upadate logs
        if RANK in (-1, 0) and self.is_rank_zero:
            self.training_logger.update_phase_end(phase, printing=True)
        
    
    def epoch_validate(self,
                       phase: str,
                       epoch: int,
                       is_training_now=True
        ):
        def _init_headline():
            header = tuple(['Epoch', 'GPU_mem'] + self.loss_names + self.metrics)
            LOGGER.info(('\n' + '%15s' * (2 + len(self.loss_names) + len(self.metrics))) % header)

        def _get_val_pbar(dloader, nb, is_rank_zero):
            if is_rank_zero:
                _init_headline()
                return TQDM(enumerate(dloader), total=nb)
            return enumerate(dloader)

        with torch.no_grad():
            val_loader = self.dataloaders[phase]
            nb = len(val_loader)
            pbar = _get_val_pbar(val_loader, nb, self.is_rank_zero)

            model = self.model_engine
            model = model.half() if self.config.half_inference else model.float()
            model.eval()

            if self.config.half_inference and self.is_rank_zero:
                LOGGER.warning('Half inference started, yet we recommend using mixed precision.')

            # validation loop
            for i, batch in pbar:
                if self.config.fast_validation_step_interval and i % self.config.fast_validation_step_interval != 0:
                    continue                    

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch_size = batch['src'].size(0)   # src is always present whether the model is seq2seq or not
                _, loss = self.model_engine(batch, return_loss=True)

                # preparing for model evaluation
                inference_batch_size = min(batch_size, self.config.fast_validation_n) if self.config.fast_validation_n else batch_size
                user_prompt = batch['user_prompt'][:inference_batch_size] if 'user_prompt' in batch else batch['src'][:inference_batch_size]
                response_gt = batch['response'][:inference_batch_size] if 'response' in batch else None
                response_pred = self.model_engine.inference(user_prompt, max_length=self.config.max_length, num_return_sequences=1, greedy=True)

                # evaluation and logging
                metric_results = self.metric_evaluation(loss, response_pred, response_gt)
                self.training_logger.update(
                    phase, 
                    epoch, 
                    self.train_cur_step if is_training_now else 0, 
                    inference_batch_size, 
                    **{'validation_loss': loss.item()}, 
                    **metric_results
                )

                # logging
                if self.is_rank_zero:
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    loss_log = [loss.item()]
                    msg = tuple([f'{epoch+1}/{self.epochs}', mem] + loss_log + [metric_results[k] for k in self.metrics])
                    if self.config.inference_result_verbose and self.is_rank_zero:
                        _init_headline()
                    pbar.set_description(('%15s' * 2 + '%15.4g' * (len(loss_log) + len(self.metrics))) % msg)

                    if self.config.inference_result_verbose and self.is_rank_zero:
                        for u, p, g in zip(user_prompt, response_pred, response_gt):
                            LOGGER.info('\n\n' + '-'*100)
                            LOGGER.info(colorstr('Prompt    : ') + u)
                            LOGGER.info(colorstr('Prediction: ') + p)
                            LOGGER.info(colorstr('GT        : ') + g)
                            LOGGER.info('-'*100 + '\n')

            # upadate logs and save model
            self.training_logger.update_phase_end(phase, printing=True)

            # gather the results of all ranks
            if self.is_ddp:
                obj = {'results': self.training_logger.validation_epoch_result, 'length': len(val_loader.dataset)}
                gathered_list = gather_objects(obj, self.is_rank_zero, self.world_size)
                if self.is_rank_zero:
                    gathered_results = calculate_gathered_results(gathered_list)
                    self.training_logger.validation_epoch_result = gathered_results

            if is_training_now:
                if self.is_rank_zero:
                    self.training_logger.save_model(self.wdir, self.model_engine)
                    self.training_logger.save_logs(self.save_dir)

                # re-freezing model for training phase
                self._freeze_model()

                            
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
                LOGGER.warning(f'Invalid key": {m}')
        
        return metric_results
    