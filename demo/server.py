import os
import re
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))

import random
import uvicorn
from sconf import Config
from threading import Thread
from fastapi import FastAPI, WebSocket
from transformers import TextIteratorStreamer


import torch

from utils import LOGGER, colorstr
from utils.filesys_utils import json_load
from trainer.build import get_model, get_peft_model



app = FastAPI()


class Chatter:
    def __init__(self, config, model_path, device, save_context=True, efficient_load=False):
        self.context = None
        self.save_context = save_context
        self.max_context_len = 256
        self.device = torch.device(device)
        self.config = config
        self.model, self.tokenizer = self.load_model(model_path, efficient_load)
        self.template = json_load('data/koalpaca_hard/templates/template1.json')


    def load_model(self, model_path, efficient_load=False):
        model, tokenizer = self._init_model(self.config)

        if efficient_load:
            model = model.to('cpu')
        
        checkpoints = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoints['model'])
        model.eval()
        del checkpoints
        torch.cuda.empty_cache()

        if efficient_load:
            return model.to(self.device), tokenizer
        return model, tokenizer
        

    def _init_model(self, config):
        # init model and tokenizer
        model, tokenizer = get_model(config, self.device)

        # init peft
        if config.peft_config_path:
            model = get_peft_model(model, config)
        else:
            LOGGER.info(f'PEFT is not applied.')

        return model, tokenizer
    

    async def print_one_by_one(self, char):
        print(char, end='', flush=True)


    def preprocess(self, message):
        instruction = re.sub(r'\n+', '\n', message)
        instructions = instruction.split('\n')
        instruction = instructions[0].strip()
        description = '\n'.join(instructions[1:]).strip()

        if description == '':
            LOGGER.info(colorstr('No discription case'))
            no_input_template = random.choice(self.template['prompt_no_input'])
            guidance_template = no_input_template.split('### Instruction')[0]
            dialogue_template = '### Instruction' + no_input_template.split('### Instruction')[-1]
            user_prompt = guidance_template + dialogue_template.format(instruction=instruction) if self.context == None else dialogue_template.format(instruction=instruction)
        else:
            LOGGER.info(colorstr('Discription case'))
            self.context = None
            template = random.choice(self.template['prompt_input'])
            user_prompt = template.format(instruction=instruction, input=description)
        
        user_prompt_tokens = torch.tensor(self.tokenizer.encode(user_prompt), dtype=torch.long).to(self.device).unsqueeze(0)

        return user_prompt_tokens
    

    def _init_generate_kwargs(self, message, src_tok, attention_mask):
        do_sample = True
        if '번역' in message or '요약' in message:
            do_sample = False
        
        generate_kwargs = {
                'input_ids': src_tok,
                'attention_mask': attention_mask,
                'min_length': 10,
                'max_length': 512,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'do_sample': do_sample,
                'top_k': 5,
                'top_p': 0.95,
                'temperature': 1,
                'no_repeat_ngram_size': 3,
                'repetition_penalty': 1.5,
                'early_stopping': True,
                'use_cache': True,
                'streamer': self.streamer,
            }
        
        return generate_kwargs
    

    async def generate(self, websocket):
        async for message in websocket.iter_text():
            self.streamer = TextIteratorStreamer(self.tokenizer)
            src_tok = self.preprocess(message)
            src_tok = src_tok if self.context == None else torch.cat((self.context, src_tok), dim=1)
            attention_mask = torch.ones_like(src_tok).to(self.device)
            response = []
            generate_kwargs = self._init_generate_kwargs(message, src_tok, attention_mask)
            
            t = Thread(target=self.model.model.generate, kwargs=generate_kwargs)
            t.daemon = True
            t.start()

            print('Response: ')
            for i, char in enumerate(self.streamer):
                # when i == 0, prompt will be printed
                if i > 0:
                    response.append(char)
                    
                    fin_char = self.pp(char)
                    await self.print_one_by_one(fin_char)
                    await websocket.send_text(fin_char)

                    if self.tokenizer.eos_token in char:
                        break

            response = ' '.join(''.join(response).split())
            if not self.tokenizer.eos_token in response:
                response += self.tokenizer.eos_token

            response_token = torch.tensor(self.tokenizer.encode(response), dtype=torch.long).unsqueeze(0).to(self.device)
            src_tok = torch.cat((src_tok, response_token), dim=1)
            if self.save_context:
                self.process_context(src_tok)

            print('\n', '-'*10, '\n\n')


    async def greedy_generate(self, websocket):
        async for message in websocket.iter_text():
            src_tok = self.preprocess(message)
            src_tok = src_tok if self.context == None else torch.cat((self.context, src_tok), dim=1)

            print('Response: ')
            for _ in range(256):
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

                await self.print_one_by_one(char)
                await websocket.send_text(char)

            if self.save_context:
                self.process_context(src_tok)

            print('\n', '-'*10, '\n\n')
        
    
    def process_context(self, context):
        self.context = self.tokenizer.encode(self.tokenizer.decode(context[0].tolist()) + '\n\n')
        self.context = torch.tensor(self.context, dtype=torch.long).unsqueeze(0).to(self.device)

        while self.context.size(1) > self.max_context_len:
            LOGGER.info(colorstr('Context is trimming...'))
            context_text = self.tokenizer.decode(self.context[0].tolist())
            context_text = self.tokenizer.eos_token.join(context_text.split(self.tokenizer.eos_token)[1:])
            context_text = context_text.lstrip()
            self.context = torch.tensor(self.tokenizer.encode(context_text), dtype=torch.long).unsqueeze(0).to(self.device)
        
        LOGGER.info(colorstr('\n\n\n___Context____'))
        print(self.tokenizer.decode(self.context[0].tolist()))


    def pp(self, text):
        if self.tokenizer.eos_token in text:
            text = text.replace(self.tokenizer.eos_token, '')
        
        return text


model_dir = '/home/junmin/nas/members/junmin/Documents/model/llm_easy/llm_test3'
config = Config(os.path.join(model_dir, 'args.yaml'))
model_path = os.path.join(model_dir, 'weights/model_epoch:1_metric_best.pt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_greedy = True
save_context = False
efficient_load = False
chatter = Chatter(config, model_path, device, save_context, efficient_load)



@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if is_greedy:
        await chatter.greedy_generate(websocket)
    else:
        await chatter.generate(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)