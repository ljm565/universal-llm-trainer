import random
import asyncio

import torch

from trainer.build import get_model, get_peft_model
from utils import LOGGER, colorstr
from utils.filesys_utils import json_load




class Chatter:
    def __init__(self, config, model_path, device):
        self.device = torch.device(device)
        self.config = config
        self.model, self.tokenizer = self._init_model(self.config)
        checkpoints = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoints['model'])
        self.model.eval()
        self.template = json_load('data/koalpaca_easy/templates/template1.json')


    def _init_model(self, config):
        # init model and tokenizer
        model, tokenizer = get_model(config, self.device)

        # init peft
        if config.peft_config_path:
            model = get_peft_model(model, config)
        else:
            LOGGER.info(f'PEFT is not applied.')

        return model, tokenizer
    

    def generate_custom(self, src_tok, max_length, num_return_sequences=1, greedy=False):
        print('Response: ')
        for _ in range(max_length):
            attention_mask = torch.ones_like(src_tok).to(self.device)
            logit = self.model.model(
                input_ids=src_tok,
                attention_mask=attention_mask,
            ).logits
            src_tok = torch.cat((src_tok, torch.argmax(logit[:, -1], dim=-1).unsqueeze(0)), dim=1)
            char = self.tokenizer.decode([src_tok[0, -1].item()])
            
            if char == self.tokenizer.eos_token:
                break
            if char == self.tokenizer.pad_token:
                break

            asyncio.run(self.print_one_by_one(char))
        return src_tok
    

    async def print_one_by_one(self, char):
        print(char, end='', flush=True)
        await asyncio.sleep(0.1)


    def do_chat(self):
        LOGGER.info(colorstr('Start chatting...'))
        LOGGER.info(f"You can {colorstr('ignore description')} by pressing Enter.")
        LOGGER.info(f"Press {colorstr('Ctrl+C')} to exit.")
        LOGGER.warning(colorstr('red', 'Only alpaca style is supported.\n'))
        warmup = True
        
        with torch.no_grad():   
            # chat
            while True:
                if warmup:
                    warmup = False
                    dummy_input = '안녕하세요'
                    dummy_input_tokens = torch.tensor(self.tokenizer.encode(dummy_input), dtype=torch.long).to(self.device).unsqueeze(0)
                    self.model.generate(dummy_input_tokens, max_length=2, num_return_sequences=1, greedy=True)
                    LOGGER.info('warmig up end\n')
                    continue
                    
                instruction = input('Instruction: ')
                description = input('Description: ')
                if not description:
                    template = random.choice(self.template['prompt_no_input'])
                    user_prompt = template.format(instruction=instruction)
                else:
                    template = random.choice(self.template['prompt_input'])
                    user_prompt = template.format(instruction=instruction, input=description)
                
                # tokenize
                user_prompt_tokens = torch.tensor(self.tokenizer.encode(user_prompt), dtype=torch.long).to(self.device).unsqueeze(0)
                # response = self.model.generate(user_prompt_tokens, max_length=256, num_return_sequences=1, greedy=True)
                response = self.generate_custom(user_prompt_tokens, max_length=256, num_return_sequences=1, greedy=True)

                # decode
                response = response[0].tolist()
                response = self.tokenizer.decode(response)
                response = response.split(self.template['response_split'])[-1]
                response = self.output_pp(response)
                response = ' '.join(response.split())

                # print(f"response: {response}")
                print('\n', '-'*50, '\n')

    
    def output_pp(self, response):
        if self.tokenizer.eos_token in response:
            response = response.split(self.tokenizer.eos_token)[0]
        if '<|endoftext|>' in response:
            response = response.split('<|endoftext|>')[0]
        if '###' in response:
            response = response.split('###')[0]
        if '#' in response:
            response = response.split('#')[0]
        return response   