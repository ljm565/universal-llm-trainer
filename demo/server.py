import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))

import uvicorn
import argparse
from sconf import Config
from fastapi import FastAPI, WebSocket

import torch

from tools import Chatter
from utils.training_utils import choose_proper_resume_model



app = FastAPI()

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir', type=str, required=False)
parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
parser.add_argument('-t', '--template_path', type=str, required=True)
parser.add_argument('-d', '--device', type=str, default='0', required=False)
parser.add_argument('-c', '--config', type=str, required=False)
parser.add_argument('--is_greedy', action='store_true', required=False)
parser.add_argument('--efficient_load', action='store_true', required=False)
parser.add_argument('--save_context', action='store_true', required=False)
args = parser.parse_args()

# init config path
if not args.model_dir:
    assert args.config is not None, 'Please provide resume model directory or config path'
config = Config(os.path.join(args.model_dir, 'args.yaml')) if args.model_dir else Config(args.config)
if 'is_rank_zero' not in config:
    config.is_rank_zero = True
if 'training_stage' not in config:
    config.training_stage = 0

model_path = choose_proper_resume_model(args.model_dir, args.load_model_type) if args.model_dir else None
device = torch.device('cpu') if args.device == 'cpu' else torch.device(f'cuda:{args.device}')
chatter = Chatter(config, model_path, device, args.template_path, args.save_context, args.efficient_load, args.is_greedy)



@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await chatter.generate(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)