# LLM Trainer
## Introduction
This repository is designed to make it easy for anyone to tune models available on Hugging Face.
When a new model is released, anyone can easily implement a model wrapper to perform instruction-tuning and fine-tuning.
For detailed usage instructions, please refer to the description below.
<br><br><br>

## Repository Structure
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
<br><br>

## Tutorials & Documentations
1. [Getting Started](./docs/1_getting_start.md)
2. [Data Preparation](./docs/2_data_preparation.md)
3. [Training](./docs/3_training.md)
