import random

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from tools.tokenizers import KoAlpacaTokenizer
from utils import print_mem_consumption
from utils.training_utils import choose_proper_model



class KoAlpaca(nn.Module):
    def __init__(self, config, device):
        super(KoAlpaca, self).__init__()
        self.model_path = choose_proper_model(config)
        self.device = device
        self.set_bit(config.bit)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            load_in_4bit=self.is4bit,
            load_in_8bit=self.is8bit,
            torch_dtype=torch.float16 if self.is16bit else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = KoAlpacaTokenizer(config, self.model_path)

        if config.is_rank_zero:
            print_mem_consumption(self.model_path)
    

    def set_bit(self, bit):
        assert bit in [4, 8, 16, 32]
        self.is4bit, self.is8bit, self.is16bit, self.is32bit = False, False, False, False
        
        if bit == 4:
            self.is4bit = True
        elif bit == 8:
            self.is8bit = True
        elif bit == 16:
            self.is16bit = True
        else:
            self.is32bit = True
    

    def _init_criterion(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    

    def forward(self, batch, return_loss=False):
        src_tok, enc_mask, label = batch['src'], batch['src_attention_mask'], batch['label']
        output = self.model(
            input_ids=src_tok,
            attention_mask=enc_mask,
        )
        if return_loss:
            output = output.logits
            loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), label[:, 1:].reshape(-1))
            return output, loss
        return output
    

    def inference(self, src, max_length, num_return_sequences=1, greedy=False):
        if isinstance(src, str):
            src_tok = torch.tensor(self.tokenizer.encode(src), dtype=torch.long).unsqueeze(0).to(self.device)
            return self.tokenizer.decode(self.generate(src_tok, max_length, num_return_sequences, greedy)[0][src_tok.size(-1):].tolist())
        elif isinstance(src, list):
            assert all([isinstance(s, str) for s in src]), f'All elements in src should be str type'
            src_tok = [torch.tensor(self.tokenizer.encode(s), dtype=torch.long).unsqueeze(0).to(self.device) for s in src]
            return [self.tokenizer.decode(self.generate(tok, max_length, num_return_sequences, greedy)[0][tok.size(-1):].tolist()) for tok in src_tok]
        else:
            raise AssertionError('Inference input should be str or list of str')


    def generate(self, src_tok, max_length, num_return_sequences=1, greedy=False):
        attention_mask = torch.ones_like(src_tok).to(self.device)
        if greedy:
            return self.model.generate(
                input_ids=src_tok,
                attention_mask=attention_mask,
                max_length=max_length,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.model.generate(
            input_ids=src_tok,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=5,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=3,
            num_beams=2,
            early_stopping=True,
            use_cache=True,
        )