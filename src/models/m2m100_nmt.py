import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from tools.tokenizers import M2M100Tokenizer
from utils import print_mem_consumption


# BART-based NMT model
class En2KoNMT(nn.Module):
    def __init__(self, config, device):
        super(En2KoNMT, self).__init__()
        self.model_path = 'facebook/m2m100_1.2B'
        self.src_lang_code, self.trg_lang_code = config.src_lang_code, config.trg_lang_code
        self.device = device
        self.set_bit(config.bit)
        
        self.device = torch.device('cuda:0')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path, 
            load_in_4bit=self.is4bit,
            load_in_8bit=self.is8bit,
            torch_dtype=torch.float16 if self.is16bit else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = M2M100Tokenizer(self.model_path, self.src_lang_code, self.trg_lang_code)

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


    def forward(self, src, trg, return_loss=False):
        src_tok, enc_mask = src
        trg_tok, dec_mask = trg
        
        output = self.model(
            input_ids=src_tok,
            attention_mask=enc_mask,
            decoder_input_ids=trg_tok,
            decoder_attention_mask=dec_mask
        )
        if return_loss:
            output = output.logits
            loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), trg_tok[:, 1:].reshape(-1))
            return loss
        return output
    



if __name__ == '__main__':
    import sys
    sys.path.append('/home/junmin/Documents/Python/llm/src/')
    from tools.tokenizers import M2M100Tokenizer

    
    tokenizer = M2M100Tokenizer('en', 'ko')
    config = {}

    model = En2KoNMT(config, tokenizer)

    src = tokenizer.encode('Hello, my dog is cute')
    trg = tokenizer.encode('안녕하세요, 내 개는 귀여워요')
    
    src, trg = torch.LongTensor(src).unsqueeze(0), torch.LongTensor(trg).unsqueeze(0)
    model(src, trg)

