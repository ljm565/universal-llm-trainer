import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))

import uvicorn
from sconf import Config
from fastapi import FastAPI, WebSocket

import torch

from tools import Chatter



app = FastAPI()

model_dir = 'outputs/phi3/mras_en_filtered_simplified6'
template_path = '/home/junmin/Documents/Python/llm/templates/phi3_templates/template1.json'
config = Config(os.path.join(model_dir, 'args.yaml'))
model_path = os.path.join(model_dir, 'weights/model_epoch:1_step:31_last_best.pt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_greedy = True
save_context = False
efficient_load = False
chatter = Chatter(config, model_path, device, template_path, save_context, efficient_load, is_greedy)



@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await chatter.generate(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)