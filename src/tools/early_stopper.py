from utils import LOGGER



class EarlyStopper:
    def __init__(self, patience=None):
        self.best_epoch, self.best_step = 0, 0
        self.is_init = True
        self.best_high, self.best_low = 0, float('inf')
        self.patience = float('inf') if patience == None or patience < 0 else patience
        self.update_n = 0


    def __call__(self, epoch, step, high=None, low=None):
        if self.is_init:
            self.is_init = False
            self.best_high, self.bestlow = high, low
            return False
        
        # update best metrics
        is_high_updated, is_low_updated = False, False
        if self.best_high != None and high > self.best_high:
            self.best_high = high
            is_high_updated = True
        
        if self.best_low != None and low < self.best_low:
            self.best_low = low
            is_low_updated = True
        
        # patience update
        if any([is_high_updated, is_low_updated]):
           self.best_epoch, self.best_step = epoch, step
           self.update_n = 0
        else:
            self.update_n += 1

        stop = self.update_n > self.patience    # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} validations. '
                        f'Best results observed at "{self.best_epoch} epochs / {self.best_step} steps".\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `patience=300` or use `patience=None` to disable EarlyStopping.')
        return stop