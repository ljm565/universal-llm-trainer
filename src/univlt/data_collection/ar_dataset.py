import os
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Tuple

import torch
from torch.utils.data import Dataset

from univlt.config import TrainingConfig
from univlt.utils import log
from univlt.utils.filesys_utils import txt_load, json_load



class AutoregressiveDataset(Dataset):
    def __init__(
            self,
            mode: str,
            cfg: TrainingConfig,
            data: list[dict], 
            tokenizer,
            template_path=None,
            name=None
        ):
        """
        Initialize an Autoregressive dataset.

        Args:
            mode (str): Dataset split (e.g., "train", "validation", "test").
            cfg (TrainingConfig): Training configuration.
            data (list[dict]): Raw dataset samples.
            tokenizer (CustomTokenizer): Tokenizer used for encoding text.
            template_path (None): Path to the prompt template file. (Unused)
            name (None): Dataset name for logging/visualization. (Unused)
        """
        # Initialize variables
        name = 'Autoregressive' if not name else name
        self.data = data
        self.tokenizer = tokenizer
        self.__init_tokens()
        self.max_length = cfg.max_length
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
        
        # If bos_token_id is not None, use it, otherwise use pad_token_id
        if self.tokenizer.bos_token_id is not None:
            self.bos_token_id = self.tokenizer.bos_token_id
            self.bos_token = self.tokenizer.bos_token
        else:
            self.bos_token_id = self.pad_token_id
            self.bos_token = self.pad_token
    
    def get_token_len_statistics(self) -> Tuple[list[int], int, int, float]:
        """
        Calculate token length statistics for the dataset.

        Returns:
            Tuple[list, int, int, float]: A tuple containing the list of token lengths, maximum length, minimum length, and average length.
        """
        max_n = 200000
        interv = len(self.data) // max_n if len(self.data) > max_n else 1
        if interv > 1:
            log(f"Length of {len(self.data)} is too long. Approximately {len(self.data) // interv} samples will be used to calculate statistics.", level='warning')
        length = [len(self.make_ar_data(i)[0]) for i in tqdm(range(0, len(self.data), interv))]
        max_length = max(length)
        min_length = min(length)
        avg_length = sum(length) / len(length)
        return length, max_length, min_length, avg_length
    

    def make_ar_data(self, idx: int) -> Tuple[list[int], str]:
        """
        Prepare a formatted prompt and corresponding token IDs for a given data sample.

        Args:
            idx (int): Index of the data sample to process.

        Returns:
            Tuple[list[int], str]: A tuple containing the list of input token IDs and formatted prompt string.
        """
        single_data = self.data[idx]

        # Assume the data has the single key 'text' for user prompt
        formatted_prompt = self.bos_token + single_data['text'][0] + self.eos_token

        full_prompt_tokens = self.tokenizer.encode(formatted_prompt)

        return full_prompt_tokens, formatted_prompt
        

    def _pad(self, data: list[int], max_length: int, pad_token_id: int) -> Tuple[list[int], int]:
        """
        Pad the input data to a specified maximum length using a given padding token ID.

        Args:
            data (list[int]): List of token IDs to be padded.
            max_length (int): Maximum length to pad the data to.
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
    

    def __getitem__(self, idx: int):
        full_prompt_token, formatted_prompt = self.make_ar_data(idx)
        
        # padding
        full_prompt_token, data_len = self._pad(
            data=full_prompt_token,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
        )
        attention_mask, _ = self._pad(
            data=self.get_mask(data_len),
            max_length=self.max_length,
            pad_token_id=0,
        )

        label = deepcopy(full_prompt_token)

        # When pad_token_id == eos_token_id, padding positions use EOS tokens
        # set to ignore_index so CrossEntropyLoss(ignore_index=ignore_index) excludes them from loss
        if self.pad_token_id == self.tokenizer.eos_token_id:
            label[data_len:self.max_length] = [self.ignore_index] * (self.max_length - data_len)
        
        assert len(full_prompt_token) == len(attention_mask) == len(label) == self.max_length, \
            f'Length of full_prompt_token, attention_mask, label are not same: {len(full_prompt_token)}, {len(attention_mask)}, {len(label)}'

        return {'src': torch.tensor(full_prompt_token, dtype=torch.long), 'src_attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long),
                'formatted_prompt': formatted_prompt}
    

    def __len__(self):
        return self.length