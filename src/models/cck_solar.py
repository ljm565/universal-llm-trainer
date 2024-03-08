import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from utils import LOGGER, colorstr



class CCKSolar:
    def __init__(self, bit, eval=False):
        self.set_bit(bit)
        self.model_path = 'JaeyeonKang/CCK-v1.0.0-DPO'
        self.device = torch.device('cuda:0')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            load_in_4bit=self.is4bit,
            load_in_8bit=self.is8bit,
            torch_dtype=torch.float16 if self.is16bit else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if eval:
            self.model.eval()
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        LOGGER.info(f'{colorstr("LDCC-SOLAR")} accounts for {colorstr(mem)} of GPU memory.')
    
    
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

    
    def ask(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        print(inputs)
        print(self.tokenizer.tokenize(text))
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=False))








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
   bit = 4
   model = CCKSolar(bit)

   text = "[INST]블랙홀에 대해 설명해줘[/INST]"
   model.ask(text)