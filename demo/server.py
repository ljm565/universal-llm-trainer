import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))

import re
import random
import uvicorn
from sconf import Config
from fastapi import FastAPI, WebSocket

import torch

from utils import LOGGER
from utils.filesys_utils import json_load
from trainer.build import get_model, get_peft_model



app = FastAPI()


class Chatter:
    def __init__(self, config, model_path, device):
        self.device = torch.device(device)
        self.config = config
        self.model, self.tokenizer = self._init_model(self.config)
        checkpoints = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoints['model'])
        self.model.eval()
        self.template = json_load('data/koalpaca_hard/templates/template1.json')


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
            LOGGER.info('no discription')
            template = random.choice(chatter.template['prompt_no_input'])
            user_prompt = template.format(instruction=instruction)
        else:
            LOGGER.info('discription')
            template = random.choice(chatter.template['prompt_input'])
            user_prompt = template.format(instruction=instruction, input=description)
        
        user_prompt_tokens = torch.tensor(chatter.tokenizer.encode(user_prompt), dtype=torch.long).to(chatter.device).unsqueeze(0)

        return user_prompt_tokens
        

    async def generate(self, websocket):
        async for message in websocket.iter_text():
            src_tok = self.preprocess(message)

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


model_dir = 'outputs/llm_hard/llm_test/'
config = Config(os.path.join(model_dir, 'args.yaml'))
model_path = os.path.join(model_dir, 'weights/model_epoch:1_step:37491_metric_best.pt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
chatter = Chatter(config, model_path, device)


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await chatter.generate(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)