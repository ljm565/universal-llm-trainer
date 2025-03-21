# Universal LLM Trainer


### Recent updates 📣
* *March 2025 (v1.5.2)*: Universal LLM trainer does not support KoPolyglot and KoGemma and, GPU memory usage during model training has been improved.
* *March 2025 (v1.5.1)*: Universal LLM trainer does not support unnecessary fucntions (e.g. NMT, translator).
* *February 2025 (v1.5.0)*: Universal LLM trainer has supported LLaMA 2 template. Please refer to the **tempaltes** folder.
* *November 2024 (v1.4.9)*: Universal LLM trainer has supported FSDP training.

&nbsp;



## Overview 📚
This repository is designed to make it easy for anyone to tune models available on Hugging Face.
When a new model is released, anyone can easily implement a model wrapper to perform instruction-tuning and fine-tuning.
For detailed usage instructions, please refer to the description below.
* Universal LLM trainer supports full-training.
* Universal LLM trainer supports LoRA fine-tuning.
* Universal LLM trainer supports QLoRA fine-tuning.
* Universal LLM trainer supports DDP and FSDP training strategies.

&nbsp;


### GPU Memory and training speed
Below is an example of the memory requirements and training speed for different models.

> [!NOTE]
> Training conditions: 
> - Environments: Ubuntu 22.04.4 LTS
> - Batch size: 2
> - Sequnece length: 8,192 (Without padding, fully filled tokens)
> - Model type: torch.bfloat16
> - Optimizer: torch.optim.AdamW
> - w/ Gradient checkpointing (Test was done with both "torch" and "Hugging Face" gradient checkpointing methods)
> - Gradient accumulation: 
>   - Full fine-tuning: 1
>   - LoRA: 32

| Model | Tuning Method | GPU | Peak Mem. (Model Mem.) | Sec/batch |
|:- |-:|-:|-:|-:|
| Llama 3.1 8B    | Full | H100 x 1  | 78 GiB (16 GiB)    | 4.7    |
| Llama 3.1 8B    | LoRA | H100 x 1  | 36 GiB (16 GiB)    | 6.4    |
| Llama 3 8B      | Full | H100 x 1  | 78 GiB (16 GiB)    | 4.7    |
| Llama 3 8B      | LoRA | H100 x 1  | 36 GiB (16 GiB)    | 6.4    |
| Llama 2 13B *   | Full | H100 x 2  | 56 GiB (25.5 GiB)  | 9.5    |  
| Llama 2 13B     | LoRA | H100 x 1  | 43 GiB (25.5 GiB)  | 9.8    |
| Gemma 2 9B *    | Full | H100 x 2  | 74 GiB (18 GiB)    | 12.6   |
| Gemma 2 9B      | LoRA | H100 x 1  | 60 GiB (18 GiB)    | 12.9   |
| Gemma 7B *      | Full | H100 x 2  | 60 GiB (18 GiB)    | 8.7    |   
| Gemma 7B        | LoRA | H100 x 1  | 51 GiB (17 GiB)    | 9.5    |
| Phi3-mini (3.8B)| Full | H100 x 1  | 40 GiB (8 GiB)     | 4.0    |
| Phi3-mini (3.8B)| LoRA | H100 x 1  | 17 GiB (8 GiB)     | 5.0    |
* : FSDP training + 32 gradient accumuation.

&nbsp;

<!-- ## Repository Structure
This repository is structured as follows.
```
├── config
│   └── *.yaml
├── config_lora
│   └── *.yaml
│
├── data
│   └── ${DATA_NAME}
│       └── ${DATA_NAME}.pkl
│
├── demo
│   ├── front
│   │   └── design
│   └── server.py
│
├── docker
│   ├── Dockerfile
│   └── requirements.txt
│
├── src
│   ├── data_collection             # Datasets wrappers
│   ├── models                      # Model wrappers
│   ├── run
│   │   ├── chat.py                 # The entry point of simple chat demo for trained LLM model
│   │   ├── train_deepspeed.py      # The entry point of deepspeed LLM training
│   │   ├── train.py                # The entry point of LLM training
│   │   └── validation.py           # The entry point of evaluation of trained LLM model
│   ├── task
│   ├── tools
│   │   ...
│   │   └── tokenizers              # LLM tokenizer wrappers
│   ├── trainer
│   │   ├── build.py
│   │   ├── trainer_deepspeed.py    # Deepspeed training trainer
│   │   └── trainer.py              # Training trainer
│   └── utils
│
└── templates                       # LLM instruction templates
```
<br><br> -->

## Tutorials & Documentations
1. [Getting Started](./docs/1_getting_start.md)
2. [Data Preparation](./docs/2_data_preparation.md)
3. [Training](./docs/3_training.md)
