import os
import random
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from utils import log
from utils.filesys_utils import txt_load, json_load



class AutoregressiveDataset(Dataset):
    def __init__(self,
                 mode,
                 config,
                 data, 
                 tokenizer,
                 template_dir=None,
                 name=None):
        # init
        name = 'Autoregressive' if not name else name
        self.data = data
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # read data and template
        template_paths = [p for p in filter(lambda x: x.startswith('template'), os.listdir(template_dir))]
        assert all([p.endswith('.txt') or p.endswith('.json') for p in template_paths]), f'Invalid template file format in {template_dir}, only possible format is .txt or .json'
        self.templates = ['\n'.join(txt_load(os.path.join(template_dir, p))) if p.endswith('.txt') \
                                else json_load(os.path.join(template_dir, p)) for p in template_paths]

        # params
        self.max_length = config.max_length
        self.add_bos = config.add_bos_token_when_response_start
        self.add_eos = config.add_eos_token_when_response_end
        self.verbose = config.data_verbose
        self.length = len(self.data)

        # calculate statistics
        if config.is_rank_zero and self.verbose:
            save_dir = os.path.join(config.save_dir, 'vis_data')
            os.makedirs(save_dir, exist_ok=True)

            log(f'Calculating statistics of {name} data...')
            src_l, src_max, src_min, src_avg = self.get_token_len_statistics(self.data)
            msg = f'{name} dataset: max={src_max}, min={src_min}, avg={src_avg}'
            log(msg)
            
            # save histograms
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.hist(src_l, bins=100)
            ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
            ax.tick_params(axis='y', labelsize=15)
            ax.tick_params(axis='x', labelsize=15)
            plt.title(f'{name} dataset', fontdict=20)
            plt.xlabel('Length of samples', fontsize=20)
            plt.ylabel('Number of samples', fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{name}_{mode}_data_hist.png'), dpi=300)

    
    def get_token_len_statistics(self, data):
        """
        Args:
            data (list): list of str.
        """
        max_n = 200000
        interv = len(data) // max_n if len(data) > max_n else 1
        if interv > 1:
            log(f"Length of {len(data)} is too long. Approximately {len(data) // interv} samples will be used to calculate statistics.", level='warning')
        length = [len(self.make_ar_data(i)[0]) for i in tqdm(range(0, len(data), interv))]
        max_length = max(length)
        min_length = min(length)
        avg_length = sum(length) / len(length)
        return length, max_length, min_length, avg_length
    

    def make_ar_data(self, idx):
        single_data = self.data[idx]
        template = random.choice(self.templates)
        template = random.choice(template['prompt_no_input'])
        
        instruction = single_data['instruction'][0]
        full_prompt = template.format(instruction=instruction)
        full_prompt_tokens = self.tokenizer.encode(full_prompt)

        return full_prompt_tokens, full_prompt
        

    def _pad(self, data, max_length, pad_token_id, bos_token=None, eos_token=None, return_data_len=False, bos_masking=False):
        # add bos and eos token
        if bos_token:
            data = [self.tokenizer.pad_token_id] + data if bos_masking else [self.tokenizer.bos_token_id] + data
        if eos_token:
            data.append(self.tokenizer.eos_token_id)
        
        # calculate data length
        data = data if len(data) <= max_length else data[:max_length]
        data_len = len(data)

        # padding
        data = data + [pad_token_id] * (max_length - len(data))
        if return_data_len:
            return data, data_len
        return data
    

    @staticmethod
    def get_mask(token_length):
        return [1] * token_length
    

    def __getitem__(self, idx):
        full_prompt_token, full_prompt = self.make_ar_data(idx)
        
        # padding
        full_prompt_token, data_len = self._pad(
            data=full_prompt_token,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            bos_token=self.tokenizer.bos_token_id if self.add_bos and self.tokenizer.bos_token_id else None,
            eos_token=self.tokenizer.eos_token_id if self.add_eos and self.tokenizer.eos_token_id else None,
            return_data_len=True
        )
        attention_mask = self._pad(self.get_mask(data_len), self.max_length, 0)

        if self.add_bos:
            full_prompt = self.tokenizer.bos_token + full_prompt
        if self.add_eos:
            full_prompt = full_prompt + self.tokenizer.eos_token

        label = deepcopy(full_prompt_token)        

        return {'src': torch.tensor(full_prompt_token, dtype=torch.long), 'src_attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long),
                'user_prompt': full_prompt}
    

    def __len__(self):
        return self.length