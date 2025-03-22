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
                 template_dir=None,
                 name=None):
        # init
        name = 'ARC'
        self.data = self.__normalize_format(data)
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

    
    def __normalize_format(self, data):
        """
        To normalize ARC benchmark dataset for LLM training.
        You can modify these function, but please stick the output formats: 'instruction', 'input', and 'output'.

        Args:
            data (dataset): Hugging Face dataset class type.
        """
        normalized_data = []
        system_message = 'You are an LLM who finds the correct answer from the given options.'
        instruction = 'Here is a question you need to solve, along with answer choices:\n\n'
        
        for single_data in data:
            choices = ''.join([f'{l}. {c}\n' for l, c in zip(single_data['choices']['label'], single_data['choices']['text'])])
            question = f"Question: {single_data['question']}\nChoices:\n{choices}"
            answer = f"The answer is {single_data['answerKey']}!"

            normalized_data.append(
                {'input': [system_message],
                 'instruction': [instruction + question],
                 'output': [answer]}
            )

        return normalized_data
    
    
    def get_token_len_statistics(self, data):
        """
        Args:
            data (list): list of str.
        """
        max_n = 200000
        interv = len(data) // max_n if len(data) > max_n else 1
        if interv > 1:
            LOGGER.warning(f"Length of {len(data)} is too long. Approximately {len(data) // interv} samples will be used to calculate statistics.")
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
                is_last = i == len(instructions) - 1
                response_end = '' if is_last else self.tokenizer.eos_token
                user_prompt = guidance_template + dialogue_template.format(instruction=instruction) if i == 0 else dialogue_template.format(instruction=instruction)
                response = response.strip() + response_end
                
                if self.tokenizer.sep_token:
                    user_prompt = user_prompt + self.tokenizer.sep_token
                    response = response if is_last else response + '\n\n'

                    all_tokens = self.tokenizer.encode(user_prompt + response)
                    sel_token_loc = all_tokens.index(self.tokenizer.sep_token_id)
                    user_prompt_tokens = all_tokens[:sel_token_loc+1]
                    response_tokens = all_tokens[sel_token_loc+1:]

                    if is_last:
                        final_user_prompt = full_prompt + user_prompt

                    full_prompt += user_prompt + response
                    full_prompt_tokens += user_prompt_tokens + response_tokens
                    label += [self.pad_token_id] * len(user_prompt_tokens) + response_tokens

                else:
                    user_prompt_tokens = self.tokenizer.encode(user_prompt)
                    response_tokens = self.tokenizer.encode(response)

                    only_response_tokens_l = len(response_tokens)
                    response = response if is_last else response + '\n\n'
                    final_response_tokens = self.tokenizer.encode(response)
                    new_line_token_l = len(final_response_tokens) - only_response_tokens_l

                    if is_last:
                        final_user_prompt = full_prompt + user_prompt
                    
                    full_prompt += user_prompt + response
                    full_prompt_tokens += user_prompt_tokens + final_response_tokens
                    label += [self.pad_token_id] * len(user_prompt_tokens) + response_tokens + [self.pad_token_id] * new_line_token_l
        else:
            template = random.choice(template['prompt_input'])
            for i, (instruction, response) in enumerate(zip(instructions, responses)):
                is_last = i == len(instructions) - 1
                response_end = '' if is_last else self.tokenizer.eos_token
                user_prompt = template.format(instruction=instruction, input=single_data['input'][0]) if i == 0 else dialogue_template.format(instruction=instruction)
                response = response.strip() + response_end

                if self.tokenizer.sep_token:
                    user_prompt = user_prompt + self.tokenizer.sep_token
                    response = response if is_last else response + '\n\n'

                    all_tokens = self.tokenizer.encode(user_prompt + response)
                    sel_token_loc = all_tokens.index(self.tokenizer.sep_token_id)
                    user_prompt_tokens = all_tokens[:sel_token_loc+1]
                    response_tokens = all_tokens[sel_token_loc+1:]

                    if is_last:
                        final_user_prompt = full_prompt + user_prompt

                    full_prompt += user_prompt + response
                    full_prompt_tokens += user_prompt_tokens + response_tokens
                    label += [self.pad_token_id] * len(user_prompt_tokens) + response_tokens
                
                else:
                    user_prompt_tokens = self.tokenizer.encode(user_prompt)
                    response_tokens = self.tokenizer.encode(response)

                    only_response_tokens_l = len(response_tokens)
                    response = response if is_last else response + '\n\n'
                    final_response_tokens = self.tokenizer.encode(response)
                    new_line_token_l = len(final_response_tokens) - only_response_tokens_l

                    if is_last:
                        final_user_prompt = full_prompt + user_prompt

                    full_prompt += user_prompt + response
                    full_prompt_tokens += user_prompt_tokens + final_response_tokens
                    label += [self.ignore_index] * len(user_prompt_tokens) + response_tokens + [self.pad_token_id] * new_line_token_l
                        
        # sanity check
        assert len(full_prompt_tokens) == len(label), \
            f'Length of full_prompt_tokens, label are not same: {len(full_prompt_tokens)}, {len(label)}'
        
        for f, l in zip(full_prompt_tokens, label):
            assert f == l or l == self.ignore_index, f'Full prompt and label are not same: {f}, {l}'
        
        return full_prompt_tokens, label, final_user_prompt, response


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
                LOGGER.warning("⚠️ All texts are invalid. Randomly select one text.")
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

