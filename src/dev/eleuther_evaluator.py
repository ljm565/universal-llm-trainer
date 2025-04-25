
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict, TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch



class EleutherEvaluator:
    def __init__(self, config):
        pass


if __name__ == '__main__':
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    device = torch.device('cuda:7')
    cache_dir = '/nfs_data_storage/huggingface/'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Eleuther model wrapper
    model_wrapper = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device,
        max_length=8196,
        dtype=torch.bfloat16,
    )

    # Task initialization
    tasks = ["mmlu"]
    task_manager = TaskManager(include_path=None)
    task_dict = get_task_dict(tasks, task_manager)


    output = evaluate(
        model_wrapper,
        task_dict,
        limit=1,
    )

    print(output)
