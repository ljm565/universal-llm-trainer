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