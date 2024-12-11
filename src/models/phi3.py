import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from tools.tokenizers import Phi3Tokenizer
from utils import print_mem_consumption, logger
from utils.training_utils import init_model_config, choose_proper_model



class Phi3(nn.Module):
    def __init__(self, config, device):
        super(Phi3, self).__init__()
        self.model_path = choose_proper_model(config)
        self.device = device
        self.load_unnecessary_half = config.load_unnecessary_half
        self.is_rank_zero = config.is_rank_zero
        self.set_bit(config.bit, config.training_stage)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            device_map=self.device,
            low_cpu_mem_usage=True,
            **init_model_config(config, self.load16bit)
        )

        if config.gradient_checkpointing:
            logger(self, 'Gradient checkpointing will be applied')
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()

        # freezing proper layers
        self.freeze_layers(config.training_stage)

        # 4, 8bit model automatically loads neccesaries to 32bit
        if self.load16bit:
            self.mapping_neccessary_32bit()

        self.tokenizer = Phi3Tokenizer(config, self.model_path)
        if hasattr(self.tokenizer, 'resized'):
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger(self, 'Model word embedding is resized to match the tokenizer')

        print_mem_consumption(self, self.model_path)
    

    def set_bit(self, bit, training_stage):
        assert bit in [4, 8, 16, 32]
        self.is4bit, self.is8bit, self.is16bit, self.is32bit = False, False, False, False
        
        if training_stage in [1, 2, 3, 4]:
            if bit == 16:
                self.is16bit = True
            else:
                self.is32bit = True

            logger('Training stage 1, 2, 3, 4 automatically loads model in 32bit or 16bit')

        else:
            if bit == 4:
                self.is4bit = True
                self.load_unnecessary_half = False
                logger(self, 'Model is loaded in 4bit')
            elif bit == 8:
                self.is8bit = True
                self.load_unnecessary_half = False
                logger(self, 'Model is loaded in 8bit')
            elif bit == 16:
                self.is16bit = True
                logger(self, 'Model is loaded in 16bit')
            else:
                self.is32bit = True
                logger(self, 'Model is loaded in 32bit')

        self.load16bit = True if self.is16bit or self.load_unnecessary_half else False

    
    def mapping_neccessary_32bit(self):
        for param in self.model.parameters():
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
    

    def _init_criterion(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    

    def forward(self, batch, return_loss=False, output_hidden_states=False):
        src_tok, enc_mask, label = batch['src'], batch['src_attention_mask'], batch['label']
        output = self.model(
            input_ids=src_tok,
            attention_mask=enc_mask,
            output_hidden_states=output_hidden_states,
        )
        if return_loss:
            loss = self.criterion(output.logits[:, :-1, :].reshape(-1, output.logits.size(-1)), label[:, 1:].reshape(-1))
            return output, loss
        return output
    

    def inference(self, src, max_length, num_return_sequences=1, greedy=False, max_time=None, synced_gpus=False):
        if isinstance(src, str):
            src_tok = torch.tensor(self.tokenizer.encode(src), dtype=torch.long).unsqueeze(0).to(self.device)
            if src_tok.size(1) >= max_length:
                return src
            return self.tokenizer.decode(self.generate(src_tok, max_length, num_return_sequences, greedy, max_time, synced_gpus)[0][src_tok.size(-1):].tolist())
        elif isinstance(src, list):
            assert all([isinstance(s, str) for s in src]), f'All elements in src should be str type'
            src_tok = [torch.tensor(self.tokenizer.encode(s), dtype=torch.long).unsqueeze(0).to(self.device) for s in src]
            return [self.tokenizer.decode(self.generate(tok, max_length, num_return_sequences, greedy, max_time, synced_gpus)[0][tok.size(-1):].tolist()) if tok.size(1) < max_length else src[i] for i, tok in enumerate(src_tok)]
        else:
            raise AssertionError('Inference input should be str or list of str')


    def generate(self, src_tok, max_length, num_return_sequences=1, greedy=False, max_time=None, synced_gpus=False):
        attention_mask = torch.ones_like(src_tok).to(self.device)
        if greedy:
            return self.model.generate(
                input_ids=src_tok,
                attention_mask=attention_mask,
                max_length=max_length,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_time=max_time,
                synced_gpus=synced_gpus,
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
            max_time=max_time,
            synced_gpus=synced_gpus,
        )
    
    def freeze_layers(self, stage):
        if stage == 1:
            logger(self, 'Freezing all layers except for word embeddings')
            
            for name, param in self.model.named_parameters():
                if 'embed' in name:
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
        elif stage == 2:
            logger(self, 'Freezing all layers except for the lm_head')
            
            for name, param in self.model.named_parameters():
                if 'lm_head' in name:
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        elif stage == 3:
            logger(self, 'Freezing all layers except for word embeddings and lm_head')
            
            for name, param in self.model.named_parameters():
                if 'embed' in name or 'lm_head' in name:
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        elif stage == 4:
            logger(self, 'Unfreezing all layers except for word embeddings and lm_head')
            
            for name, param in self.model.named_parameters():
                if 'embed' in name or 'lm_head' in name:
                    param.requires_grad = False
                else:
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True

