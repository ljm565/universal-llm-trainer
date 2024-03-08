import os
import re
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from utils import LOGGER, colorstr
from utils.filesys_utils import txt_load, json_load



class ARCDataset(Dataset):
    def __init__(self,
                 mode,
                 config,
                 data, 
                 tokenizer,
                 template_dir=None):
        # init
        name = 'ARC'
        self.flags = ['question', 'choices.text', 'choices.label', 'answerKey']
        self.data = data
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id if config.pad_token_id is None else config.pad_token_id
        
        # read data and template
        self.responses = [form.replace('\\n', '\n') for form in txt_load(os.path.join(template_dir, 'response.txt'))]
        self.instructions = [form.replace('\\n', '\n') for form in txt_load(os.path.join(template_dir, 'instruction.txt'))]
        template_paths = [p for p in filter(lambda x: x.startswith('template'), os.listdir(template_dir))]
        assert all([p.endswith('.txt') or p.endswith('.json') for p in template_paths]), f'Invalid template file format in {template_dir}, only possible format is .txt or .json'
        self.templates = ['\n'.join(txt_load(os.path.join(template_dir, p))) if p.endswith('.txt') \
                                else json_load(os.path.join(template_dir, p)) for p in template_paths]

        # params
        self.max_length = config.max_length
        self.verbose = config.data_verbose
        self.length = len(self.data)

        # calculate statistics
        if self.verbose:
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
            plt.savefig(os.path.join(save_dir, f'{mode}_data_hist.png'))

    
    def get_token_len_statistics(self, data):
        """
        Args:
            data (list): list of str.
        """
        max_n = 200000
        interv = len(data) // max_n if len(data) > max_n else 1
        if interv > 1:
            LOGGER.warning(f'Length of {colorstr(len(data))} is too long. Approximately {colorstr(len(data) // interv)} samples will be used to calculate statistics.')
        length = [len(self.tokenizer.encode(self.construct_instruction(i)[0])) for i in tqdm(range(0, len(data), interv))]
        max_length = max(length)
        min_length = min(length)
        avg_length = sum(length) / len(length)
        return length, max_length, min_length, avg_length


    def construct_instruction(self, idx):
        single_data = self.data[idx]
        template = random.choice(self.templates)
        response = self.random_choice(self.responses, single_data)
        instruction = self.random_choice(self.instructions, single_data)

        # mapping instruction, response and template
        response = single_data['answerKey'] if not response else self._mapping_data(response, single_data, 'response')
        instruction = single_data['question'] if not instruction else self._mapping_data(instruction, single_data, 'instruction')
        template = template.format(instruction=instruction, response=response, eos_token=self.tokenizer.eos_token)

        # find location to be masked
        query_end_loc = 0
        template = template.split('<<<masking_area>>>')
        if len(template) != 2:
            return template, query_end_loc
        query_end_loc = len(self.tokenizer.encode(template[0]))
        template = template[0] + template[1]
        return template, query_end_loc


    def random_choice(self, texts, data):
        indices = list(range(len(texts)))
        while 1:
            if len(indices) == 0:
                LOGGER.warning(f'All texts are invalid. Randomly select one text.')
                return None
            
            idx = random.choice(indices)
            if self._check_valid_format(texts[idx], data):
                return texts[idx]
            indices.remove(idx)


    @staticmethod
    def _check_valid_format(text, data):
        matches = re.findall(r'\{([^}]+)\}', text)      # find all texts inside the braces
                
        for match in matches:
            tmp = data
            for key in match.split('.'):
                if key not in tmp:
                    return False
                tmp = tmp[key]
        
        return True
    

    @staticmethod
    def _mapping_data(text, data, instruction_type):
        tmp_dict = {}
        matches = re.findall(r'\{([^}]+)\}', text)      # find all texts inside the braces
        
        for match in matches:
            tmp = data
            for key in match.split('.'):
                tmp = tmp[key]
            tmp_dict[match] = tmp
        
        if instruction_type == 'response':
            text = text.format(**tmp_dict)
        elif instruction_type == 'instruction':
            choices = ''
            for i, (l, t) in enumerate(zip(tmp_dict['choices.label'], tmp_dict['choices.text'])):
                if i == len(tmp_dict['choices.label']) - 1:
                    choices += f'{l}. {t}'
                else:
                    choices += f'{l}. {t}\n'
            text = text.replace('{question}', tmp_dict['question'])
            text = text.replace('{choices.label} {choices.text}', choices)
        else:
            AssertionError(f'Invalid instruction type: {instruction_type}')
        
        return text
        

    def _pad(self, data, max_length, pad_token_id):
        """
        Args:
            data (list): list of token.
            max_length (int): max length of sentence.
        """
        data = data if len(data) <= max_length else data[:max_length]
        data = data + [pad_token_id] * (max_length - len(data))
        return data
    

    @staticmethod
    def get_mask(tokens):
        return [1] * len(tokens)


    def __getitem__(self, idx):
        template, query_end_loc = self.construct_instruction(idx)
        
        # tokenize and pad
        template = self.tokenizer.encode(template)
        # query, response = template[:query_end_loc], template[query_end_loc:]
        src_mask = self.get_mask(template)

        template = self._pad(template, self.max_length,  self.pad_token_id)
        src_mask = self._pad(src_mask, self.max_length, 0)

        return {'src': torch.tensor(template, dtype=torch.long), 'src_attention_mask': torch.tensor(src_mask, dtype=torch.long),
                'query_end_loc': query_end_loc}
                # 'query': torch.tensor(query, dtype=torch.long), 'response': torch.tensor(response, dtype=torch.long)}
    

    def __len__(self):
        return self.length
    


def huggingface_arc_generator(single_data, templates, responses, instructions, tokenizer):
    def _mapping_data(text, data, instruction_type):
        tmp_dict = {}
        matches = re.findall(r'\{([^}]+)\}', text)      # find all texts inside the braces
        for match in matches:
            tmp = data
            for key in match.split('.'):
                tmp = tmp[key]
            tmp_dict[match] = tmp
        if instruction_type == 'response':
            text = text.format(**tmp_dict)
        elif instruction_type == 'instruction':
            choices = ''
            for i, (l, t) in enumerate(zip(tmp_dict['choices.label'], tmp_dict['choices.text'])):
                if i == len(tmp_dict['choices.label']) - 1:
                    choices += f'{l}. {t}'
                else:
                    choices += f'{l}. {t}\n'
            text = text.replace('{question}', tmp_dict['question'])
            text = text.replace('{choices.label} {choices.text}', choices)
        else:
            AssertionError(f'Invalid instruction type: {instruction_type}')
        return text
    
    def random_choice(texts, data):
        indices = list(range(len(texts)))
        while 1:
            if len(indices) == 0:
                LOGGER.warning(f'All texts are invalid. Randomly select one text.')
                return None
            idx = random.choice(indices)
            if _check_valid_format(texts[idx], data):
                return texts[idx]
            indices.remove(idx)

    def _check_valid_format(text, data):
        matches = re.findall(r'\{([^}]+)\}', text)      # find all texts inside the braces
        for match in matches:
            tmp = data
            for key in match.split('.'):
                if key not in tmp:
                    return False
                tmp = tmp[key]
        return True
    
    def _construct_instruction():
        template = random.choice(templates)
        response = random_choice(responses, single_data)
        instruction = random_choice(instructions, single_data)

        # mapping instruction, response and template
        response = single_data['answerKey'] if not response else _mapping_data(response, single_data, 'response')
        instruction = single_data['question'] if not instruction else _mapping_data(instruction, single_data, 'instruction')
        template = template.format(instruction=instruction, response=response, eos_token=tokenizer.eos_token)

        # find location to be masked
        query_end_loc = 0
        template = template.split('<<<masking_area>>>')
        if len(template) != 2:
            return template, query_end_loc
        query_end_loc = len(tokenizer.encode(template[0]))
        template = template[0] + template[1]
        return template, query_end_loc
    
    # for k in data.keys():

    template, query_end_loc = _construct_instruction()
    template = tokenizer(template)
    del template['token_type_ids']
    
    return template

