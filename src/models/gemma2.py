import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

from tools.tokenizers import GemmaTokenizer
from utils import print_mem_consumption, logger
from utils.func_utils import instantiate
from utils.training_utils import init_model_config, choose_proper_model



class Gemma2(nn.Module):
    def __init__(self, config, device):
        super(Gemma2, self).__init__()
        # Initialize environment settings
        self.is_rank_zero = config.is_rank_zero
        self.del_logits = config.del_logits_after_forward
        self._model_path = choose_proper_model(config)
        self.bit = self.__set_bit(config.bit)
        self.device = device

        # Initialize model and training settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_path, 
            device_map=self.device if not config.fsdp_train else None,      # Does not need to pre-define device_map for FSDP training
            low_cpu_mem_usage=True,
            torch_dtype=instantiate(torch, self.bit) if isinstance(self.bit, str) else torch.float32,
            **init_model_config(config)
        )
        self.__set_gradient_checkpointing(config)   # Gradient checkpointing setting.
        self.tokenizer = GemmaTokenizer(config, self._model_path)
        if hasattr(self.tokenizer, 'resized'):
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger(self, 'Model word embedding is resized to match the tokenizer')
        
        # Freezing proper layers
        self.freeze_layers(config.training_stage)
        print_mem_consumption(self, self._model_path)
        

    def __set_bit(self, bit):
        if isinstance(bit, int):
            assert bit in [4, 8, 16, 32]
            logger(self, f'Model is loaded in {bit}-bit')
            if bit == 32:
                bit = 'float32'
            elif bit == 16:
                bit = 'bfloat16'
                logger(self, 'Model dytpe is automatically set to torch.bfloat16')
        elif isinstance(bit, str):
            logger(self, f'Model dytpe is torch.{bit}')
        return bit

    
    def __set_gradient_checkpointing(self, config):
        if config.gradient_checkpointing.activate:
            if config.gradient_checkpointing.checkpoint_type.lower() == 'torch_checkpoint':
                logger(self, 'Torch gradient checkpointing will be applied.')
                auto_wrap_policy=ModuleWrapPolicy({Gemma2DecoderLayer})
                apply_activation_checkpointing(self.model, auto_wrap_policy=auto_wrap_policy)
            else:
                if config.gradient_checkpointing.checkpoint_type.lower() == 'hf_checkpoint':
                    logger(self, 'Hugging Face gradient checkpointing will be applied.')
                else:
                    logger(self, 'Invalid checkpoint type. Hugging Face gradient checkpointing will be applied.', 'warning')
                self.model.enable_input_require_grads()
                self.model.gradient_checkpointing_enable()

    
    def mapping_neccessary_32bit(self):
        for param in self.model.parameters():
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
    

    def init_criterion(self):
        ignore_index = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id else -100
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    

    def forward(self, batch, return_loss=False, output_hidden_states=False):
        src_tok, enc_mask, label = batch['src'], batch['src_attention_mask'], batch['label']
        output = self.model(
            input_ids=src_tok,
            attention_mask=enc_mask,
            output_hidden_states=output_hidden_states,
        )
        if return_loss:
            loss = self.criterion(output.logits[:, :-1, :].reshape(-1, output.logits.size(-1)), label[:, 1:].reshape(-1))
            if self.del_logits:
                del output
                output = None
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
                use_cache=False if synced_gpus else True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_time=max_time,
                do_sample=False,
                top_p=1,
                temperature=1,
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
                if 'embed_tokens' in name:
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
        elif stage == 2:
            logger(self, 'Freezing all layers except for the lm_head')

            for name, param in self.model.named_parameters():
                param.requires_grad = False

            for param in self.model.lm_head.parameters():
                param.data = param.data.to(torch.float32)
                param.requires_grad = True

        
        elif stage == 3:
            logger(self, 'Freezing all layers except for word embeddings and lm_head')

            for name, param in self.model.named_parameters():
                if 'embed_tokens' in name:
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            for param in self.model.lm_head.parameters():
                param.data = param.data.to(torch.float32)
                param.requires_grad = True
        
        elif stage == 4:
            logger(self, 'Unfreezing all layers except for word embeddings and lm_head')
            
            for name, param in self.model.named_parameters():
                if 'embed_tokens' in name:
                    param.requires_grad = False
                else:
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True
            
            for param in self.model.lm_head.parameters():
                param.requires_grad = False