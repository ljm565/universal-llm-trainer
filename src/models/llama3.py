import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from tools.tokenizers import Llama3Tokenizer
from utils import print_mem_consumption, logger
from utils.training_utils import init_model_config, choose_proper_model

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from utils.training_utils import set_default_dtype


import sys
import mmap
from functools import partial
from collections import OrderedDict
from typing import Any, Dict, Generator, Optional
from torch._subclasses.fake_tensor import FakeTensorConverter, FakeTensorMode



_use_low_cpu_ram = False

def _register_reparametrize_state_dict_hooks(
    module: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    offload_to_cpu: bool = True,
):
    """
    Register the reparametrize state dict hooks to the module and its submodules.

    This function is a wrapper that is meant to toggle between the low_cpu_ram
    and regular versions of the ``reparametrize_as_dtype`` state dict hooks.

    Args:
        module (nn.Module): the module to register the hooks to.
        dtype (torch.dtype): the dtype to restore the weight to. Default is ``torch.bfloat16``.
        offload_to_cpu (bool): whether to offload the restored weight to CPU. Default is ``True``.

    Raises:
        RuntimeError: If the low RAM reparametrize hook is used on Windows or an incompatible torch version.
    """
    if _use_low_cpu_ram:
        if sys.platform == "win32":
            # mmap.MAP_SHARED is not supported on Windows but this change targets colab.
            raise RuntimeError(
                "Low RAM reparametrize_as_dtype_state_dict_post_hook is not supported on Windows."
            )
        else:
            hook = _low_ram_reparametrize_as_dtype_state_dict_post_hook
    else:
        hook = reparametrize_as_dtype_state_dict_post_hook
    module._register_state_dict_hook(
        partial(hook, dtype=dtype, offload_to_cpu=offload_to_cpu)
    )

def reparametrize_as_dtype_state_dict_post_hook(
    model: nn.Module,
    state_dict: Dict[str, Any],
    *args: Any,
    dtype: torch.dtype = torch.bfloat16,
    offload_to_cpu: bool = True,
    **kwargs: Any,
):
    """
    A state_dict hook that replaces NF4 tensors with their restored
    higher-precision weight and optionally offloads the restored weight to CPU.
    Use this hook to avoid increased peak GPU memory usage during checkpoint
    save when training with QLoRA.

    This function is meant to be used with PyTorch's ``nn.Module._register_state_dict_hook``, i.e.

    >>> m = MyModule()
    >>> m._register_state_dict_hook(reparametrize_as_dtype_state_dict_post_hook)

    If the hook is registered per the above process, this hook will be called _after_ the module's
    ``state_dict`` method is called. The hook will replace all ``NF4Tensor`` instances by unquantizing
    them to the original dtype, and optionally offload the restored weight to CPU.

    Args:
        model (nn.Module): the model to take ``state_dict()`` on
        state_dict (Dict[str, Any]): the state dict to modify
        *args (Any): Unused args passed when running this as a state_dict hook.
        dtype (torch.dtype): the dtype to restore the weight to. Default is ``torch.bfloat16``.
        offload_to_cpu (bool): whether to offload the restored weight to CPU. Default is ``True``.
        **kwargs (Any): Unused keyword args passed when running this as a state_dict hook.
    """
    for k, v in state_dict.items():
        # if isinstance(v, NF4Tensor):
        if isinstance(v, torch.uint8):
            state_dict[k] = v.to(dtype)
            if offload_to_cpu:
                state_dict[k] = state_dict[k].cpu()


def _low_ram_reparametrize_as_dtype_state_dict_post_hook(
    model: nn.Module,
    state_dict: Dict[str, Any],
    *args: Any,
    dtype: torch.dtype = torch.bfloat16,
    offload_to_cpu: bool = True,
    **kwargs: Any,
):
    """
    A state_dict hook that replaces NF4 tensors with their restored
    higher-precision weight and optionally offloads the restored weight to CPU.
    Use this hook to avoid increased peak GPU memory usage during checkpoint
    save when training with QLoRA.

    This hook is similar to ``reparametrize_as_dtype_state_dict_post_hook`` but uses
    FakeTensor and mmap(2) to avoid CPU OOM on colab.

    This function is meant to be used with PyTorch's ``nn.Module._register_state_dict_hook``, i.e.

    >>> m = MyModule()
    >>> m._register_state_dict_hook(reparametrize_as_dtype_state_dict_post_hook)

    If the hook is registered per the above process, this hook will be called _after_ the module's
    ``state_dict`` method is called. The hook will replace all ``NF4Tensor`` instances by unquantizing
    them to the original dtype, and optionally offload the restored weight to CPU.

    Args:
        model (nn.Module): the model to take ``state_dict()`` on
        state_dict (Dict[str, Any]): the state dict to modify
        *args (Any): Unused args passed when running this as a state_dict hook.
        dtype (torch.dtype): the dtype to restore the weight to. Default is ``torch.bfloat16``.
        offload_to_cpu (bool): whether to offload the restored weight to CPU. Default is ``True``.
        **kwargs (Any): Unused keyword args passed when running this as a state_dict hook.
    """
    # Create a state dict of FakeTensors that matches the state_dict
    mode = FakeTensorMode()
    converter = FakeTensorConverter()
    fake_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if isinstance(v, NF4Tensor):
            fake_state_dict[k] = converter.from_real_tensor(mode, v).to(dtype)
        else:
            fake_state_dict[k] = converter.from_real_tensor(mode, v)

        if offload_to_cpu:
            fake_state_dict[k] = fake_state_dict[k].cpu()

    # Create a state_dict on disk with space reserved for storage bytes
    # Then load with mmap and MAP_SHARED (can writeback to disk file)
    dest_state_dict_path = "/tmp/fake_state_dict.pt"
    with torch.serialization.skip_data(materialize_fake_tensors=True):
        torch.save(fake_state_dict, dest_state_dict_path)
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        dest_state_dict = torch.load(dest_state_dict_path, mmap=True, weights_only=True)

    # Do D2H and upcast one by one and since dest_state_dict is backed by mmap --> won't OOM
    # even when there is no swap space (e.g. colab)
    for k in state_dict.keys():
        if isinstance(state_dict[k], NF4Tensor):
            dest_state_dict[k].copy_(state_dict[k].to(dtype))
        else:
            dest_state_dict[k].copy_(state_dict[k])

    # In place update original state_dict object. Although the private state dict
    # post hook supports out of place behavior, the semantic actually buggy. We eventually want
    # to use the public state_dict post hook which does not support out of place behavior.
    for k in state_dict.keys():
        state_dict[k] = dest_state_dict[k]



class Llama3(nn.Module):
    def __init__(self, config, device):
        super(Llama3, self).__init__()
        self.model_path = choose_proper_model(config)
        self.device = device
        self.load_unnecessary_half = config.load_unnecessary_half
        self.is_rank_zero = config.is_rank_zero
        self.set_bit(config.bit, config.training_stage)

        # with set_default_dtype(torch.bfloat16):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map=self.device,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            **init_model_config(config, self.load16bit)
        )
        
        # _register_reparametrize_state_dict_hooks(self.model)

        if config.gradient_checkpointing:
            logger(self, 'Gradient checkpointing will be applied')
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()
            # auto_wrap_policy=ModuleWrapPolicy({LlamaDecoderLayer})
            # apply_activation_checkpointing(self.model, auto_wrap_policy=auto_wrap_policy)

        # freezing proper layers
        self.freeze_layers(config.training_stage)

        # 4, 8bit model automatically loads neccesaries to 32bit
        if self.load16bit:
            self.mapping_neccessary_32bit()

        self.tokenizer = Llama3Tokenizer(config, self.model_path)
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

            logger(self, 'Training stage 1, 2, 3, 4 automatically loads model in 32bit or 16bit')

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
        ignore_index = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id else -100
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    

    def forward(self, batch, return_loss=False, output_hidden_states=False):
        src_tok, enc_mask, label = batch['src'], batch['src_attention_mask'], batch['label']
        
        src_tok = torch.randint(0, 10001, (2, 8192), dtype=torch.long).to(self.device)
        label = torch.randint(0, 10001, (2, 8192), dtype=torch.long).to(self.device)
        
        output = self.model(
            input_ids=src_tok,
            attention_mask=enc_mask,
            output_hidden_states=output_hidden_states,
        )
        if return_loss:
            loss = self.criterion(output.logits[:, :-1, :].reshape(-1, output.logits.size(-1)), label[:, 1:].reshape(-1))
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
                use_cache=True,
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

