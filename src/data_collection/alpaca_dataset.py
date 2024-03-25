import os
import re
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from utils import LOGGER, colorstr
from utils.filesys_utils import txt_load, json_load



class AlpacaDataset(Dataset):
    def __init__(self,
                 mode,
                 config,
                 data, 
                 tokenizer,
                 template_dir=None,
                 name=None):
        # init
        name = 'Alpaca' if not name else name
        self.data = data
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.generate_prompt = self.generate_prompt_multi_turn if config.is_multi_turn else self.generate_prompt_single_turn
        
        # read data and template
        template_paths = [p for p in filter(lambda x: x.startswith('template'), os.listdir(template_dir))]
        assert all([p.endswith('.txt') or p.endswith('.json') for p in template_paths]), f'Invalid template file format in {template_dir}, only possible format is .txt or .json'
        self.templates = ['\n'.join(txt_load(os.path.join(template_dir, p))) if p.endswith('.txt') \
                                else json_load(os.path.join(template_dir, p)) for p in template_paths]

        # params
        self.max_length = config.max_length
        self.add_eos = config.add_eos_token_when_respose_end
        self.verbose = config.data_verbose
        self.length = len(self.data)

        # calculate statistics
        if config.is_rank_zero and self.verbose:
            save_dir = os.path.join(config.save_dir, 'vis_data')
            os.makedirs(save_dir, exist_ok=True)

            LOGGER.info(f'Calculating statistics of {name} data...')
            src_l, src_max, src_min, src_avg = self.get_token_len_statistics(self.data)
            msg = f'{name} dataset: max={src_max}, min={src_min}, avg={src_avg}'
            LOGGER.info(msg)
            
            # save histograms
            plt.figure(figsize=(10, 10))
            plt.hist(src_l, bins=100)
            plt.title(f'{name} dataset')
            plt.xlabel('Length of samples')
            plt.ylabel('Number of samples')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{name}_{mode}_data_hist.png'))

    
    def get_token_len_statistics(self, data):
        """
        Args:
            data (list): list of str.
        """
        max_n = 200000
        interv = len(data) // max_n if len(data) > max_n else 1
        if interv > 1:
            LOGGER.warning(f"Length of {colorstr('yellow', len(data))} is too long. Approximately {colorstr('yellow', len(data) // interv)} samples will be used to calculate statistics.")
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
        label = [self.pad_token_id] * len(user_prompt_tokens) + response_tokens

        assert len(full_prompt_tokens) == len(label), \
            f'Length of full_prompt_tokens, attention_mask, label are not same: {len(full_prompt_tokens)}, {len(label)}'
        
        return full_prompt_tokens, label, user_prompt, response
    

    def generate_prompt_multi_turn(self, idx):
        single_data = self.data[idx]
        template = random.choice(self.templates)
        if len(single_data['instruction']) < 2:
            return self.generate_prompt_single_turn(idx)
        
        # multi-turn sanity check
        full_prompt, full_prompt_tokens, label = '', [], []
        responses = single_data['output']
        instructions = single_data['instruction']
        assert len(responses) == len(instructions), f'Length of instruction and response are not same: {len(instructions)}, {len(responses)}'

        # conversation template
        no_input_template = random.choice(template['prompt_no_input'])
        guidance_template = no_input_template.split('### Instruction')[0]
        dialogue_template = '### Instruction' + no_input_template.split('### Instruction')[-1]
        
        if len(single_data['input']) == 0:
            for i, (instruction, response) in enumerate(zip(instructions, responses)):
                response_end = '' if i == len(instructions) - 1 else self.tokenizer.eos_token + '\n\n'
                user_prompt = guidance_template + dialogue_template.format(instruction=instruction) if i == 0 else dialogue_template.format(instruction=instruction)
                response = response + response_end if response[-1] != '\n' else response[:-1] + response_end
                
                user_prompt_tokens = self.tokenizer.encode(user_prompt)
                response_tokens = self.tokenizer.encode(response)
                
                full_prompt += user_prompt + response     
                label += [self.pad_token_id] * len(user_prompt_tokens) + response_tokens
        else:
            template = random.choice(template['prompt_input'])
            for i, (instruction, response) in enumerate(zip(instructions, responses)):
                response_end = '' if i == len(instructions) - 1 else self.tokenizer.eos_token + '\n\n'
                user_prompt = template.format(instruction=instruction, input=single_data['input'][0]) if i == 0 else dialogue_template.format(instruction=instruction)
                response = response + response_end if response[-1] != '\n' else response[:-1] + response_end
            
                user_prompt_tokens = self.tokenizer.encode(user_prompt)
                response_tokens = self.tokenizer.encode(response)
                
                full_prompt += user_prompt + response
                label += [self.pad_token_id] * len(user_prompt_tokens) + response_tokens

        full_prompt_tokens = self.tokenizer.encode(full_prompt)

        assert len(full_prompt_tokens) == len(label), \
            f'Length of full_prompt_tokens, attention_mask, label are not same: {len(full_prompt_tokens)}, {len(label)}'
        
        return full_prompt_tokens, label, user_prompt, response
        

    def _pad(self, data, max_length, pad_token_id, add_eos=False, return_data_len=False):
        if add_eos and data[-1] != pad_token_id and len(data) < max_length:
            data.append(self.tokenizer.eos_token_id)
        
        data = data if len(data) <= max_length else data[:max_length]
        data_len = len(data)
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
        full_prompt_token, data_len = self._pad(full_prompt_token, self.max_length,  self.pad_token_id, self.add_eos, True)
        label = self._pad(label, self.max_length, self.pad_token_id, self.add_eos)
        attention_mask = self._pad(self.get_mask(data_len), self.max_length, 0)
        
        assert len(full_prompt_token) == len(attention_mask) == len(label) == self.max_length, \
            f'Length of template, attention_mask, label are not same: {len(full_prompt_token)}, {len(attention_mask)}, {len(label)}'
        
        return {'src': torch.tensor(full_prompt_token, dtype=torch.long), 'src_attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long),
                'user_prompt': user_prompt, 'response': response}
    

    def __len__(self):
        return self.length