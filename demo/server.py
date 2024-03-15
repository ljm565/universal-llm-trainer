import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))

import json
import random
from sconf import Config
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from src.trainer import Chatter
from src.utils import LOGGER


model_dir = 'outputs/llm_easy/llm_test3/'
config = Config(os.path.join(model_dir, 'args.yaml'))
model_path = os.path.join(model_dir, 'weights/model_epoch:1_loss_best.pt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
chatter = Chatter(config, model_path, device)


class Server(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        message = json.loads(self.rfile.read(length))
        instruction, description = message['instruction'], message['description']

        if description == '':
            LOGGER.info('no discription')
            template = random.choice(chatter.template['prompt_no_input'])
            user_prompt = template.format(instruction=instruction)
        else:
            LOGGER.info('discription')
            template = random.choice(chatter.template['prompt_input'])
            user_prompt = template.format(instruction=instruction, input=description)
        
        # tokenize and post
        user_prompt_tokens = torch.tensor(chatter.tokenizer.encode(user_prompt), dtype=torch.long).to(chatter.device).unsqueeze(0)
        response = chatter.generate_custom(user_prompt_tokens, max_length=256, num_return_sequences=1, greedy=True)
        response = chatter.tokenizer.decode(response[0].tolist())
        response = response.split(chatter.template['response_split'])[-1]
        response = chatter.output_pp(response)
        response = ' '.join(response.split())

        message['response'] = response
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        self.wfile.write(json.dumps(message).encode())



def run(server_class=HTTPServer, handler_class=Server, port=8502):    
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)

    LOGGER.info(f'Starting inference module on port {port}...')
    httpd.serve_forever()



if __name__ == '__main__':
    run()