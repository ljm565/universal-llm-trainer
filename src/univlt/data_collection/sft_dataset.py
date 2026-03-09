import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from univlt.config import TrainingConfig
from univlt.utils import log, colorstr
from univlt.utils.filesys_utils import json_load



class SFTDataset(Dataset):
    """
    Dataset class for Supervised Fine-Tuning (SFT).

    This dataset formats raw samples into prompt-response pairs using
    predefined templates, tokenizes them, and prepares labels suitable
    for autoregressive supervised training. It supports both single-turn
    and multi-turn data formats.
    """
    def __init__(
            self,
            mode: str,
            cfg: TrainingConfig,
            data: list[dict], 
            tokenizer,
            template_path: str,
            name: Optional[str] = None
        ):
        """
        Initialize an SFT dataset.

        Args:
            mode (str): Dataset split (e.g., "train", "validation", "test").
            cfg (TrainingConfig): Training configuration.
            data (list[dict]): Raw dataset samples.
            tokenizer (CustomTokenizer): Tokenizer used for encoding text.
            template_path (str): Path to the prompt template file.
            name (Optional[str], optional): Dataset name for logging/visualization.
        """
        # Initialize variables
        name = 'SFT' if not name else name
        self.data = data
        self.tokenizer = tokenizer
        self.__init_tokens()
        self.prompt_generate_func = self.generate_prompt_multi_turn if cfg.data_cfg.is_multi_turn else self.generate_prompt_single_turn
        self.max_length = cfg.max_length
        self.verbose = cfg.data_cfg.data_verbose
        self.length = len(self.data)
        
        # Load template
        self.template = json_load(template_path)

        # calculate statistics
        if cfg.is_rank_zero and self.verbose:
            save_dir = os.path.join(cfg.save_dir, 'vis_data')
            os.makedirs(save_dir, exist_ok=True)

            log(f'Calculating statistics of {name} data...')
            src_l, src_max, src_min, src_avg = self.get_token_len_statistics()
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


    def __init_tokens(self):
        """
        Initialize EOS and PAD tokens and their IDs.

        If either EOS or PAD token is missing, it is inferred from the other.
        At least one of them must exist.

        Raises:
            AssertionError: At least one of `eos_token_id` and `pad_token_id` must exist.
        """
        self.eos_token, self.eos_token_id = self.tokenizer.eos_token, self.tokenizer.eos_token_id
        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.ignore_index = -100

        if self.pad_token_id is None and self.eos_token_id is not None:
            self.pad_token_id = self.eos_token_id
            self.pad_token = self.eos_token
        elif self.pad_token_id is not None and self.eos_token_id is None:
            self.eos_token_id = self.pad_token_id
            self.eos_token = self.pad_token
        elif self.pad_token_id is None and self.eos_token_id is None:
            raise AssertionError(colorstr('red', 'Both `eos_token_id` and `pad_token_id` are not existing.'))
            
    
    def get_token_len_statistics(self) -> Tuple[int, int, int, float]:
        """
        Calculate token length statistics (max, min, average) for the dataset.

        Returns:
            Tuple[int, int, int, float]: A tuple containing the list of token lengths, maximum length, minimum length, and average length.
        """
        max_n = 200000
        interv = len(self.data) // max_n if len(self.data) > max_n else 1
        if interv > 1:
            log(f"Length of {len(self.data)} is too long. Approximately {len(self.data) // interv} samples will be used to calculate statistics.", level='warning')
        length = [len(self.prompt_generate_func(i)[0]) for i in tqdm(range(0, len(self.data), interv))]
        max_length = max(length)
        min_length = min(length)
        avg_length = sum(length) / len(length)
        return length, max_length, min_length, avg_length


    def generate_prompt_single_turn(self, idx: int) -> Tuple[list[int], list[int], str, str]:
        """
        Prepare a single-turn prompt and corresponding labels for a given data sample.

        Args:
            idx (int): Index of the data sample to process.

        Returns:
            Tuple[list[int], list[int], str, str]: A tuple containing the list of input token IDs, list of label token IDs, formatted prompt string, and response string.
        """
        single_data = self.data[idx]
        template = self.template['system_prompt_template']
        beggining_text = self.template['beginning_text']
        response_split = self.template['response_split']
        system_prompt = single_data['system_prompt'][0] if len(single_data['system_prompt']) > 0 else ''
        user_prompt = single_data['user_prompt'][0]
        response = single_data['response'][0]

        # Formatting the prompt and response
        if beggining_text:
            formatted_prompt = beggining_text + template.format(system_prompt=system_prompt, user_prompt=user_prompt, response=response)
        else:
            formatted_prompt = template.format(system_prompt=system_prompt, user_prompt=user_prompt, response=response)
        
        # Tokenization and label preparation
        prefix_text = formatted_prompt.split(response_split)[0] + response_split
        input_ids = self.tokenizer.encode(formatted_prompt)
        prefix_ids = self.tokenizer.encode(prefix_text)
        label = [self.ignore_index] * len(input_ids)
        response_start = len(prefix_ids)
        label[response_start:] = input_ids[response_start:]

        # sanity check
        assert len(input_ids) == len(label), \
            f'Length of input_ids, label are not same: {len(input_ids)}, {len(label)}'
        
        for f, l in zip(input_ids, label):
            assert f == l or l == self.ignore_index, f'Input ids and label are not same: {f}, {l}'
        
        return input_ids, label, formatted_prompt, response
    

    def generate_prompt_multi_turn(self, idx: int) -> Tuple[list[int], list[int], str, str]:
        """
        Prepare a multi-turn prompt and corresponding labels for a given data sample.

        Args:
            idx (int): Index of the data sample to process.

        Returns:
            Tuple[list[int], list[int], str, str]: A tuple containing the list of input token IDs, list of label token IDs, formatted prompt string, and the last response string.
        """
        single_data = self.data[idx]
        if len(single_data['user_prompt']) < 2:
            return self.generate_prompt_single_turn(idx)
        
        # Multi-turn sanity check
        responses = single_data['response']
        user_prompts = single_data['user_prompt']
        assert len(responses) == len(user_prompts), f'Length of user_prompt and response are not same: {len(user_prompts)}, {len(responses)}'

        # Multi-turn prompt generation
        system_template = self.template['system_prompt_template']
        user_template = self.template['user_prompt_template']
        beggining_text = self.template['beginning_text']
        response_split = self.template['response_split']
        system_prompt = single_data['system_prompt'][0] if len(single_data['system_prompt']) > 0 else ''
        user_prompts = single_data['user_prompt']
        responses = single_data['response']

        for i, (user_prompt, response) in enumerate(zip(user_prompts, responses)):
            # First turn: include system prompt and beginning text
            if i == 0:
                if beggining_text:
                    formatted_prompt = beggining_text + system_template.format(system_prompt=system_prompt, user_prompt=user_prompt, response=response)
                else:
                    formatted_prompt = system_template.format(system_prompt=system_prompt, user_prompt=user_prompt, response=response)

                # Tokenization and label preparation
                prefix_text = formatted_prompt.split(response_split)[0] + response_split
                input_ids = self.tokenizer.encode(formatted_prompt)
                prefix_ids = self.tokenizer.encode(prefix_text)
                label = [self.ignore_index] * len(input_ids)
                response_start = len(prefix_ids)
                label[response_start:] = input_ids[response_start:]
            
            # Subsequent turns: only include user prompt and response
            else:
                # Tokenization and label preparation
                one_formatted_prompt = user_template.format(user_prompt=user_prompt, response=response)
                prefix_text = one_formatted_prompt.split(response_split)[0] + response_split
                one_input_ids = self.tokenizer.encode(one_formatted_prompt)
                one_prefix_ids = self.tokenizer.encode(prefix_text)
                one_label = [self.ignore_index] * len(one_input_ids)
                one_response_start = len(one_prefix_ids)
                one_label[one_response_start:] = one_input_ids[one_response_start:]

                # Merge with previous turns
                input_ids += one_input_ids
                label += one_label
                formatted_prompt += one_formatted_prompt

        # sanity check
        assert len(input_ids) == len(label), \
            f'Length of input_ids, label are not same: {len(input_ids)}, {len(label)}'
        
        for f, l in zip(input_ids, label):
            assert f == l or l == self.ignore_index, f'Input ids and label are not same: {f}, {l}'
        
        return input_ids, label, formatted_prompt, response
        

    def _pad(self, data: list[int], max_length: int, pad_token_id: int) -> Tuple[list[int], int]:
        """
        Pad the input data to a specified maximum length using a given padding token ID.

        Args:
            data (list[int]): List of token IDs to be padded.
            max_length (int):  Maximum length to pad the data to.
            pad_token_id (int): Token ID to use for padding.

        Returns:
            Tuple[list[int], int]: A tuple containing the padded list of token IDs and the original length of the data before padding.
        """
        # calculate data length
        data = data if len(data) <= max_length else data[:max_length]
        data_len = len(data)

        # padding
        data = data + [pad_token_id] * (max_length - len(data))
        return data, data_len
    

    @staticmethod
    def get_mask(token_length: int):
        """
        Get attention mask.

        Args:
            token_length (int): Token length to do attention.

        Returns:
            list[int]: Attention mask.
        """
        return [1] * token_length
    

    def __getitem__(self, idx):
        full_prompt_token, label, formatted_prompt, response = self.prompt_generate_func(idx)
        
        # padding
        full_prompt_token, data_len = self._pad(
            data=full_prompt_token,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
        )
        label, _ = self._pad(
            data=label,
            max_length=self.max_length,
            pad_token_id=self.ignore_index,
        )
        attention_mask, _ = self._pad(
            data=self.get_mask(data_len), 
            max_length=self.max_length, 
            pad_token_id=0,
        )

        assert len(full_prompt_token) == len(attention_mask) == len(label) == self.max_length, \
            f'Length of template, attention_mask, label are not same: {len(full_prompt_token)}, {len(attention_mask)}, {len(label)}'

        return {'src': torch.tensor(full_prompt_token, dtype=torch.long), 'src_attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long),
                'formatted_prompt': formatted_prompt, 'response': response}
    

    def __len__(self):
        return self.length