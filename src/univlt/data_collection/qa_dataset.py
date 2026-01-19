import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from univlt.config import TrainingConfig
from univlt.utils import log
from univlt.utils.filesys_utils import txt_load, json_load



class QADataset(Dataset):
    def __init__(
            self,
            mode,
            cfg: TrainingConfig,
            data, 
            tokenizer,
            template_path=None,
            name=None
        ):
        # init
        name = 'QA' if not name else name
        self.data = data
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.ignore_index = self.pad_token_id if self.pad_token_id != self.tokenizer.eos_token_id else -100
        self.generate_prompt = self.generate_prompt_multi_turn if cfg.data_cfg.is_multi_turn else self.generate_prompt_single_turn
        
        # read data and template
        self.template = json_load(template_path)

        # params
        self.max_length = cfg.max_length
        self.add_bos = cfg.data_cfg.add_bos_token_when_response_start
        self.add_eos = cfg.data_cfg.add_eos_token_when_response_end
        self.verbose = cfg.data_cfg.data_verbose
        self.length = len(self.data)

        # calculate statistics
        if cfg.is_rank_zero and self.verbose:
            save_dir = os.path.join(cfg.save_dir, 'vis_data')
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
            plt.title(f'{name} dataset', fontsize=20)
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
        length = [len(self.generate_prompt(i)[0]) for i in tqdm(range(0, len(data), interv))]
        max_length = max(length)
        min_length = min(length)
        avg_length = sum(length) / len(length)
        return length, max_length, min_length, avg_length


    def generate_prompt_single_turn(self, idx):
        single_data = self.data[idx]
        template = self.template['system_prompt_template']
        response = single_data['response'][0]
        
        system_prompt = single_data['system_prompt'][0] if len(single_data['system_prompt']) > 0 else ''
        user_prompt = single_data['user_prompt'][0]
        formatted_prompt = template.format(system_prompt=system_prompt, user_prompt=user_prompt)

        formatted_prompt_tokens = self.tokenizer.encode(formatted_prompt)
        response_tokens = self.tokenizer.encode(response)
        full_prompt_tokens = formatted_prompt_tokens + response_tokens
        label = [self.ignore_index] * len(formatted_prompt_tokens) + response_tokens

        # sanity check
        assert len(full_prompt_tokens) == len(label), \
            f'Length of full_prompt_tokens, label are not same: {len(full_prompt_tokens)}, {len(label)}'
        
        for f, l in zip(full_prompt_tokens, label):
            assert f == l or l == self.ignore_index, f'Full prompt and label are not same: {f}, {l}'
        
        return full_prompt_tokens, label, formatted_prompt, response
    

    def generate_prompt_multi_turn(self, idx):
        single_data = self.data[idx]
        template = self.template['system_prompt_template']
        if len(single_data['user_prompt']) < 2:
            return self.generate_prompt_single_turn(idx)
        
        # multi-turn sanity check
        responses = single_data['response']
        user_prompts = single_data['user_prompt']
        assert len(responses) == len(user_prompts), f'Length of user_prompt and response are not same: {len(user_prompts)}, {len(responses)}'

        # conversation template
        try:
            multiturn_split = self.template['multiturn_split']
        except:
            log("Current model does not support multi-turn training", level='error')

        full_prompt_tokens, label, formatted_prompt = [], [], ''

        for i, (user_prompt, response) in enumerate(zip(user_prompts, responses)):
            is_first = i == 0
            is_last = i == len(user_prompts) - 1
            one_response = response + multiturn_split if not is_last else response

            # Processing the current turn
            one_formatted_prompt = template.format(
                system_prompt = '' if (len(single_data['system_prompt']) == 0 or not is_first) else single_data['system_prompt'][0],
                user_prompt=user_prompt
            )
            
            one_formatted_prompt_tokens = self.tokenizer.encode(one_formatted_prompt)
            one_response_tokens = self.tokenizer.encode(one_response)
            one_full_prompt_tokens = one_formatted_prompt_tokens + one_response_tokens
            one_label = [self.ignore_index] * len(one_formatted_prompt_tokens) + one_response_tokens

            # Sanity check
            assert one_full_prompt_tokens == self.tokenizer.encode(one_formatted_prompt + one_response)
            
            # Processing the entire turns
            full_prompt_tokens += one_full_prompt_tokens
            label += one_label
            formatted_prompt += self.tokenizer.decode(one_full_prompt_tokens) if not is_last else self.tokenizer.decode(one_formatted_prompt_tokens)
        
        # sanity check
        assert len(full_prompt_tokens) == len(label), \
            f'Length of full_prompt_tokens, label are not same: {len(full_prompt_tokens)}, {len(label)}'
        
        for f, l in zip(full_prompt_tokens, label):
            assert f == l or l == self.ignore_index, f'Full prompt and label are not same: {f}, {l}'
        
        return full_prompt_tokens, label, formatted_prompt, response
        

    def _pad(self, data, max_length, pad_token_id, bos_token_id=None, eos_token_id=None, return_data_len=False, bos_masking=False):
        # add bos and eos token
        if bos_token_id != None:
            data = [pad_token_id] + data if bos_masking else [bos_token_id] + data
        if eos_token_id != None:
            data.append(eos_token_id)
        
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
        full_prompt_token, label, formatted_prompt, response = self.generate_prompt(idx)
        
        # padding
        full_prompt_token, data_len = self._pad(
            data=full_prompt_token,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id if self.add_bos and self.tokenizer.bos_token_id else None,
            eos_token_id=self.tokenizer.eos_token_id if self.add_eos and self.tokenizer.eos_token_id else None,
            return_data_len=True
        )
        label = self._pad(
            data=label,
            max_length=self.max_length,
            pad_token_id=self.ignore_index,
            bos_token_id=self.tokenizer.bos_token_id if self.add_bos and self.tokenizer.bos_token_id else None,
            eos_token_id=self.tokenizer.eos_token_id if self.add_eos and self.tokenizer.eos_token_id else None,
            bos_masking=True
        )
        attention_mask = self._pad(self.get_mask(data_len), self.max_length, 0)

        if self.add_bos:
            formatted_prompt = self.tokenizer.bos_token + formatted_prompt
        if self.add_eos:
            response = response + self.tokenizer.eos_token
        
        assert len(full_prompt_token) == len(attention_mask) == len(label) == self.max_length, \
            f'Length of template, attention_mask, label are not same: {len(full_prompt_token)}, {len(attention_mask)}, {len(label)}'

        return {'src': torch.tensor(full_prompt_token, dtype=torch.long), 'src_attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long),
                'formatted_prompt': formatted_prompt, 'response': response}
    

    def __len__(self):
        return self.length