import torch



class ModelManager:
    def __init__(self):
        self.is_init = True
        self.best_lower, self.best_higher = None, None
        self.lower_candidates_weights = {
            'validation_loss': 0.6,
            'ppl': 0,
            'edit_distance': 0.4
        }
        self.higher_candidates_weights = {
            'bleu': 0.33,
            'rouge': 0.33,
            'meteor': 0.33
        }


    def save(self, model, model_path, log_data, is_rank_zero):
        model_state_dict = model.state_dict()
        if is_rank_zero:
            checkpoint = {'model': model_state_dict, 'log_data': log_data}
            torch.save(checkpoint, model_path)


    def update_best(self, epoch_log_data):
        lower_candidates = self.select_keys(epoch_log_data, higher_better=False)
        higher_candidates = self.select_keys(epoch_log_data, higher_better=True)
        cur_lower = sum([epoch_log_data[candidate] * self.lower_candidates_weights[candidate] for candidate in lower_candidates]) if len(lower_candidates) > 0 else None
        cur_higher = sum([epoch_log_data[candidate] * self.higher_candidates_weights[candidate] for candidate in higher_candidates]) if len(higher_candidates) > 0 else None

        # update best
        lower_update_flag, higher_update_flag = False, False
        if self.is_init:
            self.best_lower, self.best_higher = cur_lower, cur_higher
            self.is_init = False
            lower_update_flag = True if self.best_lower is not None else False
            higher_update_flag = True if self.best_higher is not None else False
        else:
            if (cur_lower is not None and 
                self.best_lower is not None and 
                cur_lower < self.best_lower):
                self.best_lower = cur_lower
                lower_update_flag = True

            if (cur_higher is not None and 
                self.best_higher is not None and 
                cur_higher > self.best_higher):
                self.best_higher = cur_higher
                higher_update_flag = True

        return lower_update_flag, higher_update_flag


    def select_keys(self, epoch_log_data, higher_better=True):
        if higher_better:
            candidates = self.higher_candidates_weights.keys()
            return [candidate for candidate in candidates if candidate in epoch_log_data and epoch_log_data[candidate] != None]
        else:
            candidates = self.lower_candidates_weights.keys()
            return [candidate for candidate in candidates if candidate in epoch_log_data and epoch_log_data[candidate] != None]
