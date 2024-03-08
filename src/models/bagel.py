import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from tools.tokenizers import BagelTokenizer
from utils import print_mem_consumption
from utils.training_utils import choose_proper_model


# Bagel model
class Bagel(nn.Module):
    def __init__(self, config, device):
        super(Bagel, self).__init__()
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
        self.tokenizer = BagelTokenizer(self.model_path)
        
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
    

    def make_mask(self, src):
        mask = torch.where(src==self.tokenizer.pad_token_id, 0, 1)
        return mask
    

    def forward(self, src, return_loss=False):
        src_mask = self.make_mask(src)
        output = self.model(input_ids=src, attention_mask=src_mask)
        if return_loss:
            output = output.logits
            loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), src[:, 1:].reshape(-1))
            return loss
        return output
        







# MODEL = 'LDCC/LDCC-SOLAR-10.7B'

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL, 
#     load_in_8bit=True, 
#     device_map='auto',
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL)

# for param in model.parameters():
#   param.requires_grad = False  # freeze the model - train adapters later
#   if param.ndim == 1:
#     # cast the small parameters (e.g. layernorm) to fp32 for stability
#     param.data = param.data.to(torch.float32)

# model.gradient_checkpointing_enable()  # reduce number of stored activations
# model.enable_input_require_grads()

# class CastOutputToFloat(nn.Sequential):
#   def forward(self, x): return super().forward(x).to(torch.float32)
# model.lm_head = CastOutputToFloat(model.lm_head)


# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )

# from peft import LoraConfig, get_peft_model 

# config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# model = get_peft_model(model, config)
# print_trainable_parameters(model)


# import transformers
# from datasets import load_dataset
# data = load_dataset("Abirate/english_quotes")
# data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)

# trainer = transformers.Trainer(
#     model=model, 
#     train_dataset=data['train'],
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=4, 
#         gradient_accumulation_steps=4,
#         warmup_steps=100, 
#         max_steps=200, 
#         learning_rate=2e-4, 
#         fp16=True,
#         logging_steps=1, 
#         output_dir='outputs'
#     ),
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
# )
# model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
# trainer.train()
            
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


    bit = 4
    model = Bagel(bit)