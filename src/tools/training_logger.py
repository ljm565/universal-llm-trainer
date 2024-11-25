import os
import pickle
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from .model_manager import ModelManager
from utils import LOGGER, colorstr



class TrainingLogger:
    def __init__(self, config, training=True):
        self.training = training
        self.log_data = {'step': [], 'epoch': []}
        self.log_keys = config.common + config.metrics
        self.log_data.update({k: [] for k in self.log_keys})
        self.train_batch_sizes, self.val_batch_sizes = [], []
        self.is_rank_zero = config.is_rank_zero
        self.st = 0
        if self.is_rank_zero and self.training:
            LOGGER.info(f'{colorstr("Logging data")}: {self.log_keys}')
            self.writer = SummaryWriter(log_dir=config.save_dir)
        self.model_manager = ModelManager()
        self.tensorboard_logging_interval = config.tensorboard_logging_interval
    

    def nan_value_filtering(self, values, batch_sizes):
        value_sum = sum([b*vv for b, vv in zip(batch_sizes, values) if not np.isnan(vv)])
        nan_indices = np.where(np.isnan(values))[0]
        reduce_batch_sizes = sum([batch_sizes[idx] for idx in nan_indices])
        return value_sum, reduce_batch_sizes
    

    def _update_tensorboard(self, phase, step, tag, scalar_value):
        if phase == 'train' and step % self.tensorboard_logging_interval == 0:
            self.writer.add_scalar(tag, scalar_value, step)
        else:
            self.writer.add_scalar(tag, scalar_value, step)
        self.writer.flush()


    def update(self, phase, epoch, step, batch_size, **kwargs):
        # training case
        if phase == 'train':
            self.log_data['step'].append(step)
            self.log_data['epoch'].append(epoch)
            self.train_batch_sizes.append(batch_size)
            if self.is_rank_zero:
                self._update_tensorboard(phase, step, 'epoch', epoch)
            for k in self.log_keys:
                if k in kwargs:
                    self.log_data[k].append(kwargs[k])
                    if self.is_rank_zero:
                        self._update_tensorboard(phase, step, k, kwargs[k])
                # validation log data
                else:
                    self.log_data[k].append(None)
        # validation case
        else:
            self.val_batch_sizes.append(batch_size)
            for k in self.log_keys:
                if k in kwargs:
                    # train.py case
                    try:
                        if isinstance(self.log_data[k][step], list):
                            self.log_data[k][step].append(kwargs[k])
                        else:
                            self.log_data[k][step] = [kwargs[k]]
                    # validation.py case
                    except IndexError:
                        self.log_data[k].append([kwargs[k]])


    def update_phase_end(self, phase, printing=False):
        if phase == 'train':
            train_epoch_result = {}
            for k, v in self.log_data.items():
                if k == 'train_loss':
                    # nan value processing
                    value_sum, reduce_batch_sizes = self.nan_value_filtering(v[self.st:], self.train_batch_sizes)
                    train_epoch_result[k] = value_sum / (sum(self.train_batch_sizes) - reduce_batch_sizes)
            self.st += len(self.train_batch_sizes)
            self.train_batch_sizes = []
        else:
            self.validation_epoch_result = {}
            for k, v in self.log_data.items():
                if len(v) and isinstance(v[-1], list):
                    assert len(self.val_batch_sizes) == len(v[-1])
                    # nan value processing
                    value_sum, reduce_batch_sizes = self.nan_value_filtering(v[-1], self.val_batch_sizes)
                    v[-1] = value_sum / (sum(self.val_batch_sizes) - reduce_batch_sizes)
                    self.validation_epoch_result[k] = v[-1]
                    if self.training and self.is_rank_zero:
                        self._update_tensorboard(phase, self.log_data['step'][-1], k, v[-1])
            self.val_batch_sizes = []
            
        if printing:
            result = train_epoch_result if phase == 'train' else self.validation_epoch_result
            msg = [f'{k}={v:.4f}' for k, v in result.items()]
            LOGGER.info(f"{colorstr('green', 'bold', ', '.join(msg))}\n")

    
    def delete_file(self, save_dir, flag, is_rank_zero):
        if is_rank_zero:
            file = list(filter(lambda x: flag in x, os.listdir(save_dir)))
            if len(file) > 0:
                os.remove(os.path.join(save_dir, file[0]))


    def save_model(self, save_dir, model, is_rank_zero, is_fsdp=False):
        if is_rank_zero or is_fsdp:
            if not hasattr(self, 'validation_epoch_result') or len(self.validation_epoch_result) == 0:
                LOGGER.warning(f'{colorstr("red", "No log data to save")}')
                return

            epoch, step = self.log_data['epoch'][-1], self.log_data['step'][-1]
            lower_flag, higher_flag = self.model_manager.update_best(self.validation_epoch_result)

            if lower_flag:
                self.delete_file(save_dir, 'loss', is_rank_zero)
                model_path = os.path.join(save_dir, f'model_epoch:{epoch}_step:{step}_loss_best.pt')
                self.model_manager.save(model, model_path, self.validation_epoch_result, is_rank_zero)

            if higher_flag:
                self.delete_file(save_dir, 'metric', is_rank_zero)
                model_path = os.path.join(save_dir, f'model_epoch:{epoch}_step:{step}_metric_best.pt')
                self.model_manager.save(model, model_path, self.validation_epoch_result, is_rank_zero)
            
            self.delete_file(save_dir, 'last', is_rank_zero)
            model_path = os.path.join(save_dir, f'model_epoch:{epoch}_step:{step}_last_best.pt')
            self.model_manager.save(model, model_path, self.validation_epoch_result, is_rank_zero)

    
    def save_logs(self, save_dir, is_rank_zero):
            self.delete_file(save_dir, 'log_data', is_rank_zero)
            if is_rank_zero:
                with open(os.path.join(save_dir, 'log_data.pkl'), 'wb') as f:
                    pickle.dump(self.log_data, f)