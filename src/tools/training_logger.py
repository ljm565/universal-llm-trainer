from .evaluator import Evaluator

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
        for k, v in self.tmp_log_data.items():
            if len(v) > 0:
                if self.log_data['logging_step'] == 'epoch':
                    # calculate epoch average exept for lr
                    if k in ['lr']:
                        self.log_data[k].append(self.tmp_log_data['lr'][0])
                    else:
                        self.log_data[k].append(sum([b*d for b, d in zip(self.batch_sizes, v)]) / sum(self.batch_sizes))
                else:
                    self.log_data[k] += self.tmp_log_data[k]
        
        if printing:
            msg = []
            for k, v in self.tmp_log_data.items():
                if not k in ['lr'] and len(v) > 0:
                    if self.log_data['logging_step'] == 'epoch':
                        msg.append(f'{k}={self.log_data[k][-1]:.4f}')
                    else:
                        msg.append(f'{k}={sum([b*d for b, d in zip(self.batch_sizes, v)]) / sum(self.batch_sizes):.4f}')
            LOGGER.info(colorstr('green', 'bold', ', '.join(msg)))
        self.tmp_log_data = self._init()
        self.batch_sizes = []
        

