import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))

import uvicorn
import argparse
from pathlib import Path
from sconf import Config
from fastapi import FastAPI, WebSocket

import torch

from tools import Chatter
from utils.common_utils import replace_none_value
from utils.training_utils import choose_proper_resume_model



app = FastAPI()

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir', type=str, required=False)
parser.add_argument('-a', '--adapter_path', type=str, required=False)
parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
parser.add_argument('-t', '--template_path', type=str, required=True)
parser.add_argument('--max_length', type=int, default=1024, required=False)
parser.add_argument('-d', '--device', type=str, default='0', required=False)
parser.add_argument('-c', '--config', type=str, required=False)
parser.add_argument('--is_greedy', action='store_true', required=False)
parser.add_argument('--efficient_load', action='store_true', required=False)
parser.add_argument('--save_context', action='store_true', required=False)
args = parser.parse_args()


# Sanity check
if not args.model_dir and not args.adapter_path:
    assert args.config is not None, 'Please provide resume model directory or config path'

# Load the training config file
if args.model_dir:
    config = Config(os.path.join(args.model_dir, 'args.yaml'))
elif args.adapter_path:
    config = Config(Path(args.adapter_path).parent.parent / 'args.yaml')
else:
    config = Config(args.config)
config = Config(replace_none_value(config))

if 'is_rank_zero' not in config:
    config.is_rank_zero = True
if 'training_stage' not in config:
    config.training_stage = 0

# Initialize chatter class
model_path = choose_proper_resume_model(args.model_dir, args.load_model_type) if args.model_dir else None
device = torch.device('cpu') if args.device == 'cpu' else torch.device(f'cuda:{args.device}')
chatter = Chatter(
    config=config, 
    model_path=model_path, 
    adpater_path=args.adapter_path,
    device=device, 
    template_path=args.template_path, 
    max_context_len=args.max_length,
    save_context=args.save_context, 
    efficient_load=args.efficient_load, 
    is_greedy=args.is_greedy
)

# Websocket and uvicorn loading
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await chatter.generate(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)