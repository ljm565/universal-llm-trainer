import re
import random
import asyncio
from threading import Thread
from transformers import TextIteratorStreamer

import torch

from utils import LOGGER, colorstr
from utils.filesys_utils import json_load
from trainer.build import get_model, get_peft_model




class Chatter:
    def __init__(self, config, model_path, device, template_path, save_context=True, efficient_load=False, is_greedy=False):
        self.context = None
        self.save_context = save_context
        self.is_greedy = is_greedy
        self.max_context_len = 256
        self.device = torch.device(device)
        self.config = config
        self.model, self.tokenizer = self.load_model(model_path, efficient_load)
        self.template = json_load(template_path)


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

        user_prompt = user_prompt + self.tokenizer.sep_token if self.tokenizer.sep_token else user_prompt
        user_prompt_tokens = torch.tensor(self.tokenizer.encode(user_prompt), dtype=torch.long).to(self.device).unsqueeze(0)

        return user_prompt_tokens
    

    def _init_generate_kwargs(self, message, src_tok, attention_mask, is_greedy=False):
        if is_greedy:
            return {
                'input_ids': src_tok,
                'attention_mask': attention_mask,
                'min_length': 10,
                'max_length': 512,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'do_sample': False,
                'top_k': 1,
                'early_stopping': True,
                'use_cache': True,
                'no_repeat_ngram_size': 3,
                'repetition_penalty': 1.5,
                'streamer': self.streamer,
            }

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
                'top_k': 2,
                'top_p': 0.98,
                'temperature': 0.5,
                'no_repeat_ngram_size': 3,
                'repetition_penalty': 1.5,
                'early_stopping': True,
                'use_cache': True,
                'streamer': self.streamer,
            }
        
        return generate_kwargs
    

    ########################### Below codes are used to websocket api communication ###########################
    async def generate(self, websocket):
        async for message in websocket.iter_text():
            self.streamer = TextIteratorStreamer(self.tokenizer)
            src_tok = self.preprocess(message)
            src_tok = src_tok if self.context == None else torch.cat((self.context, src_tok), dim=1)
            attention_mask = torch.ones_like(src_tok).to(self.device)
            response = []
            generate_kwargs = self._init_generate_kwargs(message, src_tok, attention_mask, self.is_greedy)
            
            t = Thread(target=self.model.model.generate, kwargs=generate_kwargs)
            t.daemon = True
            t.start()

            print('Response: ')
            for i, char in enumerate(self.streamer):
                # when i == 0, prompt will be printed
                if i > 0:
                    response.append(char)

                    if self.tokenizer.sep_token and self.tokenizer.sep_token in char:
                        continue
                    
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
    #################################################################################################
        

    ########################### Below codes are used to terminal chatting ###########################
    def generate_demo(self, message, is_greedy):
        self.streamer = TextIteratorStreamer(self.tokenizer)
        src_tok = self.preprocess(message)
        src_tok = src_tok if self.context == None else torch.cat((self.context, src_tok), dim=1)
        attention_mask = torch.ones_like(src_tok).to(self.device)
        response = []
        generate_kwargs = self._init_generate_kwargs(message, src_tok, attention_mask, is_greedy)
        
        t = Thread(target=self.model.model.generate, kwargs=generate_kwargs)
        t.daemon = True
        t.start()

        print('Response: ')
        for i, char in enumerate(self.streamer):
            # when i == 0, prompt will be printed
            if i > 0:
                response.append(char)
                
                if self.tokenizer.sep_token and self.tokenizer.sep_token in char:
                    continue

                fin_char = self.pp(char)
                asyncio.run(self.print_one_by_one(fin_char))

                if self.tokenizer.eos_token in char:
                    break

        response = ' '.join(''.join(response).split())
        if not self.tokenizer.eos_token in response:
            response += self.tokenizer.eos_token

        response_token = torch.tensor(self.tokenizer.encode(response), dtype=torch.long).unsqueeze(0).to(self.device)
        src_tok = torch.cat((src_tok, response_token), dim=1)
        if self.save_context:
            self.process_context(src_tok)

        print('\n\n')
    #################################################################################################
    

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
    

    def do_chat(self, is_greedy):
        LOGGER.info(colorstr('Start chatting...'))
        LOGGER.info(f"You can {colorstr('ignore discription')} by pressing Enter.")
        LOGGER.info(f"Press {colorstr('Ctrl+C')} to exit.")
        LOGGER.warning(colorstr('red', 'Only alpaca style is supported.\n'))

        self.model.eval()
        with torch.no_grad():   
            # chat
            while True:
                instruction = input('Instruction: ')
                discription = input('Discription: ')
                message = instruction + '\n' + discription
                self.generate_demo(message, is_greedy)
