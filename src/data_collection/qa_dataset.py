import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from utils import log
from utils.filesys_utils import txt_load, json_load



class QADataset(Dataset):
    def __init__(self,
                 mode,
                 config,
                 data, 
                 tokenizer,
                 template_dir=None,
                 name=None):
        # init
        name = 'QA' if not name else name
        self.data = data
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.ignore_index = self.pad_token_id if self.pad_token_id != self.tokenizer.eos_token_id else -100
        self.generate_prompt = self.generate_prompt_multi_turn if config.is_multi_turn else self.generate_prompt_single_turn
        
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
        template = random.choice(self.templates)
        response = single_data['output'][0]
        if len(single_data['input']) == 0:
            template = random.choice(template['prompt_no_input'])
            instruction = single_data['instruction'][0]
            user_prompt = template.format(instruction=instruction)
        else:
            template = random.choice(template['prompt_input'])
            instruction, input = single_data['instruction'][0], single_data['input'][0]
            user_prompt = template.format(instruction=instruction, input=input)

        user_prompt_tokens = self.tokenizer.encode(user_prompt)
        response_tokens = self.tokenizer.encode(response)
        full_prompt_tokens = user_prompt_tokens + response_tokens
        label = [self.ignore_index] * len(user_prompt_tokens) + response_tokens

        # sanity check
        assert len(full_prompt_tokens) == len(label), \
            f'Length of full_prompt_tokens, label are not same: {len(full_prompt_tokens)}, {len(label)}'
        
        for f, l in zip(full_prompt_tokens, label):
            assert f == l or l == self.ignore_index, f'Full prompt and label are not same: {f}, {l}'
        
        return full_prompt_tokens, label, user_prompt, response
    

    def generate_prompt_multi_turn(self, idx):
        single_data = self.data[idx]
        template = random.choice(self.templates)
        if len(single_data['instruction']) < 2:
            return self.generate_prompt_single_turn(idx)
        
        # multi-turn sanity check
        responses = single_data['output']
        instructions = single_data['instruction']
        assert len(responses) == len(instructions), f'Length of instruction and response are not same: {len(instructions)}, {len(responses)}'

        # conversation template
        input_template = random.choice(template['prompt_input'])
        no_input_template = random.choice(template['prompt_no_input'])
        try:
            multiturn_split = template['multiturn_split']
        except:
            log("Current model does not support multi-turn training", level='error')

        full_prompt_tokens, label, user_prompt = [], [], ''

        for i, (instruction, response) in enumerate(zip(instructions, responses)):
            is_first = i == 0
            is_last = i == len(instructions) - 1    # Whether the last turn or not
            one_response = response + multiturn_split if not is_last else response

            # Processing the current turn
            if is_first and len(single_data['input']) != 0:
                one_user_prompt = input_template.format(input=single_data['input'][0], instruction=instruction)
            else:
                one_user_prompt = no_input_template.format(instruction=instruction)
            one_user_prompt_tokens = self.tokenizer.encode(one_user_prompt)
            one_response_tokens = self.tokenizer.encode(one_response)
            one_full_prompt_tokens = one_user_prompt_tokens + one_response_tokens
            one_label = [self.ignore_index] * len(one_user_prompt_tokens) + one_response_tokens

            # Sanity check
            assert one_full_prompt_tokens == self.tokenizer.encode(one_user_prompt + one_response)
            
            # Processing the entire turns
            full_prompt_tokens += one_full_prompt_tokens
            label += one_label
            user_prompt += self.tokenizer.decode(one_full_prompt_tokens) if not is_last else self.tokenizer.decode(one_user_prompt_tokens)
        
        # sanity check
        assert len(full_prompt_tokens) == len(label), \
            f'Length of full_prompt_tokens, label are not same: {len(full_prompt_tokens)}, {len(label)}'
        
        for f, l in zip(full_prompt_tokens, label):
            assert f == l or l == self.ignore_index, f'Full prompt and label are not same: {f}, {l}'
        
        return full_prompt_tokens, label, user_prompt, response
        

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
        full_prompt_token, label, user_prompt, response = self.generate_prompt(idx)
        
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
            user_prompt = self.tokenizer.bos_token + user_prompt
        if self.add_eos:
            response = response + self.tokenizer.eos_token
        
        assert len(full_prompt_token) == len(attention_mask) == len(label) == self.max_length, \
            f'Length of template, attention_mask, label are not same: {len(full_prompt_token)}, {len(attention_mask)}, {len(label)}'

        return {'src': torch.tensor(full_prompt_token, dtype=torch.long), 'src_attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long),
                'user_prompt': user_prompt, 'response': response}
    

    def __len__(self):
        return self.length