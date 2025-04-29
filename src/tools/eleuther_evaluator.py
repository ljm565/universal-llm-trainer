
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict, TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

from utils import log, colorstr
from utils.filesys_utils import json_save
from utils.common_utils import instantiate



class EleutherEvaluator:
    def __init__(self, config):
        self.__device_setup(config)
        self.dtype = instantiate(torch, config.dtype)
        self.max_length = config.max_length
        self.tasks = config.tasks
        self.include_path = config.include_path
        self.limit = config.limit


    def __device_setup(self, config):
        self.device = torch.device('cpu') if config.device == 'cpu' else torch.device(f'cuda:{config.device[0]}')
        if self.device.type == 'cuda':
            torch.cuda.set_device(config.device[0])

    
    def __model_setup(self, model, tokenizer):
        model = model.to(self.device)
        model.eval()
        self.model_wrapper = HFLM(
            pretrained=model.model,
            tokenizer=tokenizer.tokenizer,
            device=self.device,
            max_length=self.max_length,
            dtype=self.dtype,
        )

    
    def model_evaluate(self, model, tokenizer, **kwargs):
        # Initialize environments
        log('Evaluation environment setting..', color=True)
        write_path = kwargs.get('write_path', None)
        self.__model_setup(model, tokenizer)
        task_manager = TaskManager(include_path=self.include_path)
        task_dict = get_task_dict(self.tasks, task_manager)

        # Start lm_eval harness evaluation
        log('Evaluation starts!', color=True)
        results = evaluate(
            self.model_wrapper,
            task_dict,
            limit=self.limit,
        )

        if write_path:
            log(f'The results file has been saved at {colorstr(write_path)}')
            json_save(write_path, results)

        return results
