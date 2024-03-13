import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from utils import LOGGER, colorstr



class NMTDataset(Dataset):
    def __init__(self, 
                 config,
                 path, 
                 tokenizer):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.verbose = config.data_verbose
        self.src_code, self.trg_code = config.src_lang_code, config.trg_lang_code

        # read csv
        mode = path[path.rfind('/')+1:path.rfind('.')]
        LOGGER.info(f'Loading {colorstr(mode)} NMT data...')
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        self.src, self.trg = df[self.src_code].tolist(), df[self.trg_code].tolist()

        # calculate statistics
        if self.verbose:
            save_dir = os.path.join(config.save_dir, 'vis_data')
            os.makedirs(save_dir, exist_ok=True)
            
            LOGGER.info(f'Calculating statistics of {colorstr(self.src_code)} sentences...')
            src_l, src_max, src_min, src_avg = self.get_token_len_statistics(self.src)
            LOGGER.info(f'Calculating statistics of {colorstr(self.trg_code)} sentences...')
            trg_l, trg_max, trg_min, trg_avg = self.get_token_len_statistics(self.trg)
            msg = f'{colorstr(self.src_code)}: max={src_max}, min={src_min}, avg={src_avg}\n{colorstr(self.trg_code)}: max={trg_max}, min={trg_min}, avg={trg_avg}'
            LOGGER.info(msg)
            
            # save histograms
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 1, 1)
            plt.hist(src_l, bins=100)
            plt.title(self.src_code)
            plt.ylabel('Number of samples')
            plt.subplot(2, 1, 2)
            plt.hist(trg_l, bins=100)
            plt.title(self.trg_code)
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
            LOGGER.warning(f"Length of {colorstr('yellow', len(data))} is too long. Approximately {colorstr('yellow', len(data) // interv)} samples will be used to calculate statistics.")
        length = [len(self.tokenizer.encode(data[i])) for i in tqdm(range(0, len(data), interv)) if isinstance(data[i], str)]
        max_length = max(length)
        min_length = min(length)
        avg_length = sum(length) / len(length)
        return length, max_length, min_length, avg_length
    

    def pad(self, data, max_length, pad_token_id):
        """
        Args:
            data (list): list of token.
            max_length (int): max length of sentence.
        """
        data = data if len(data) <= max_length else data[:max_length-1] + [self.tokenizer.eos_token_id]
        data = data + [pad_token_id] * (max_length - len(data))
        return data


    @staticmethod
    def get_mask(tokens):
        return [1] * len(tokens)


    def __getitem__(self, idx):
        src = self.tokenizer.encode(self.src[idx], self.src_code)
        trg = self.tokenizer.encode(self.trg[idx], self.trg_code)
        src_mask, trg_mask = self.get_mask(src), self.get_mask(trg)

        src, trg = self.pad(src, self.max_length, self.tokenizer.pad_token_id), self.pad(trg, self.max_length, self.tokenizer.pad_token_id)
        src_mask, trg_mask = self.pad(src_mask, self.max_length, 0), self.pad(trg_mask, self.max_length, 0)

        return {'src': torch.tensor(src, dtype=torch.long), 'src_mask': torch.tensor(src_mask, dtype=torch.long),
                'trg': torch.tensor(trg, dtype=torch.long), 'trg_mask': torch.tensor(trg_mask, dtype=torch.long)}
    

    def __len__(self):
        return len(self.src)