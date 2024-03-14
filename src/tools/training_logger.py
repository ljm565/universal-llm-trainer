import os
import pickle

from .model_manager import ModelManager
from utils import LOGGER, colorstr



class TrainingLogger:
    def __init__(self, config):
        self.log_data = {'logging_step': config.optimizer_step_criterion}
        self.log_keys = config.common + config.metrics
        self.log_data.update(self._init())
        self.tmp_log_data = self._init()
        self.batch_sizes = []
        if config.is_rank_zero:
            LOGGER.info(f'{colorstr("Logging data")}: {[k for k in self.log_data.keys()]}')
        self.model_manager = ModelManager()


    def _init(self):
        return {k: [] for k in self.log_keys}
    
    
    def update(self, batch_size, **kwargs):
        self.batch_sizes.append(batch_size)
        for k, v in kwargs.items():
            if k in self.tmp_log_data:
                self.tmp_log_data[k].append(v)
            else:
                LOGGER.warning(f'{colorstr("red", "Invalid key")}: {k}')

    
    def update_phase_end(self, printing=False):
        def _make_step_logs_to_epoch_logs(logs):
            # calculate epoch average exept for lr
            tmp_log_data_per_epoch = {}
            for k, v in logs.items():
                if len(v) > 0:
                    if k in ['lr']:
                        tmp_log_data_per_epoch[k] = [v[0]]
                    else:
                        tmp_log_data_per_epoch[k] = [sum([b*d for b, d in zip(self.batch_sizes, v)]) / sum(self.batch_sizes)]
                else:
                    tmp_log_data_per_epoch[k] = []
            return tmp_log_data_per_epoch

        self.tmp_log_data_per_epoch = _make_step_logs_to_epoch_logs(self.tmp_log_data)
        for k, v in self.tmp_log_data.items():
            if len(v) > 0:
                if self.log_data['logging_step'] == 'epoch':
                    self.log_data[k] += self.tmp_log_data_per_epoch[k]
                else:
                    self.log_data[k] += self.tmp_log_data[k]
        
        if printing:
            msg = []
            for k, v in self.tmp_log_data_per_epoch.items():
                if not k in ['lr'] and len(v) > 0:
                    msg.append(f'{k}={self.tmp_log_data_per_epoch[k][-1]:.4f}')
            LOGGER.info(f"{colorstr('green', 'bold', ', '.join(msg))}\n")

        # reset
        self.tmp_log_data = self._init()
        self.batch_sizes = []

    
    def delete_file(self, save_dir, flag):
        file = list(filter(lambda x: flag in x, os.listdir(save_dir)))
        if len(file) > 0:
            os.remove(os.path.join(save_dir, file[0]))


    def save_model(self, save_dir, epoch, model):
        if not hasattr(self, 'tmp_log_data_per_epoch'):
            LOGGER.warning(f'{colorstr("red", "No log data to save")}')
            return

        lower_flag, higher_flag = self.model_manager.update_best(self.tmp_log_data_per_epoch)

        if lower_flag:
            self.delete_file(save_dir, 'loss')
            model_path = os.path.join(save_dir, f'model_epoch:{epoch+1}_loss_best.pt')
            self.model_manager.save(model, model_path, self.tmp_log_data_per_epoch)

        if higher_flag:
            self.delete_file(save_dir, 'metric')
            model_path = os.path.join(save_dir, f'model_epoch:{epoch+1}_metric_best.pt')
            self.model_manager.save(model, model_path, self.tmp_log_data_per_epoch)
        
        self.delete_file(save_dir, 'last')
        model_path = os.path.join(save_dir, f'model_epoch:{epoch+1}_last_best.pt')
        self.model_manager.save(model, model_path, self.tmp_log_data_per_epoch)

    
    def save_logs(self, save_dir):
        self.delete_file(save_dir, 'log_data')
        with open(os.path.join(save_dir, 'log_data.pkl'), 'wb') as f:
            pickle.dump(self.log_data, f)