import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from tools.tokenizers import Llama3Tokenizer
from utils import LOGGER, print_mem_consumption, colorstr
from utils.training_utils import choose_proper_model



class Llama3(nn.Module):
    def __init__(self, config, device):
        super(Llama3, self).__init__()
        self.model_path = choose_proper_model(config)
        self.device = device
        self.load_unnecessary_half = config.load_unnecessary_half
        self.set_bit(config.bit, config.training_stage, config.is_rank_zero)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            load_in_4bit=self.is4bit,
            load_in_8bit=self.is8bit,
            torch_dtype=torch.float16 if self.load16bit else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )

        # freezing proper layers
        self.freeze_layers(config.training_stage)

        # 4, 8bit model automatically loads neccesaries to 32bit
        if self.load16bit:
            self.mapping_neccessary_32bit()

        self.tokenizer = Llama3Tokenizer(config, self.model_path)
        if hasattr(self.tokenizer, 'resized'):
            self.model.resize_token_embeddings(len(self.tokenizer))
            if config.is_rank_zero:
                LOGGER.info(colorstr('Model word embedding is resized to match the tokenizer'))

        if config.is_rank_zero:
            print_mem_consumption(self.model_path)
    

    def set_bit(self, bit, training_stage, is_rank_zero=False):
        assert bit in [4, 8, 16, 32]
        self.is4bit, self.is8bit, self.is16bit, self.is32bit = False, False, False, False
        
        if training_stage in [1, 2, 3]:
            self.is32bit = True
            self.load16bit = False
            if is_rank_zero:
                LOGGER.info(colorstr('Training stage 1, 2, 3 automatically loads model in 32bit'))
        else:
            if bit == 4:
                self.is4bit = True
            elif bit == 8:
                self.is8bit = True
            elif bit == 16:
                self.is16bit = True
                raise AssertionError('16bit is not supported yet')
            else:
                self.is32bit = True

            self.load16bit = True if self.is16bit or self.load_unnecessary_half else False

    
    def mapping_neccessary_32bit(self):
        for param in self.model.parameters():
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
    

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
    
    def freeze_layers(self, stage):
        if stage == 1:
            LOGGER.info(colorstr('Freezing all layers except for word embeddings'))
            for name, param in self.model.named_parameters():
                if 'embed' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
        elif stage == 2:
            LOGGER.info(colorstr('Freezing all layers except for the lm_head'))
            for name, param in self.model.named_parameters():
                if 'lm_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        elif stage == 3:
            LOGGER.info(colorstr('Freezing all layers except for word embeddings and lm_head'))
            for name, param in self.model.named_parameters():
                if 'embed' in name or 'lm_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        elif stage == 4:
            LOGGER.info(colorstr('Unfreezing all layers except for word embeddings and lm_head'))
            for name, param in self.model.named_parameters():
                if 'embed' in name or 'lm_head' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

