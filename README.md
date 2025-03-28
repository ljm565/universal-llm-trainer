# Universal LLM Trainer


### Recent updates 📣
* *March 2025 (v1.5.4)*: Universal LLM trainer supports **Llama 3.1 70B LoRA** training and GPU memory usage during FSDP model training has been improved.
* *March 2025 (v1.5.3)*: QLoRA test results have been added.
* *March 2025 (v1.5.2)*: Universal LLM trainer does not support KoPolyglot and KoGemma, and support Llama 2 and Gemma 1. Also, GPU memory usage during model training has been improved.
* *March 2025 (v1.5.1)*: Universal LLM trainer does not support unnecessary fucntions (e.g. NMT, translator).
* *February 2025 (v1.5.0)*: Universal LLM trainer has supported LLaMA 2 template. Please refer to the **tempaltes** folder.
* *November 2024 (v1.4.9)*: Universal LLM trainer has supported FSDP training.

&nbsp;

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
> - Environments: Ubuntu 22.04.4 LTS, torch==2.5.1, transformers==4.49.0
> - Batch size: 2
> - Sequnece length: 8,192 (Without padding, fully filled tokens)
> - Model type: torch.bfloat16
> - Optimizer: torch.optim.AdamW
> - w/ Gradient checkpointing (Tests were done with both "torch" and "Hugging Face" gradient checkpointing methods)
> - Gradient accumulation: 
>   - Full fine-tuning: 1
>   - LoRA: 32

| Model | Tuning Method | GPU | Peak Mem. (Model Mem.) | Sec/step |
|:- |-:|-:|-:|-:|
| [Llama 3.1 8B](config/llm_llama3.1_full.yaml)                 | Full   | H100 x 1  | 78 GiB (16 GiB)       | 4.7    |
| [Llama 3.1 8B](config/llm_llama3.1_lora.yaml)                 | LoRA   | H100 x 1  | 36 GiB (16 GiB)       | 6.4    |
| [Llama 3.1 8B](config/llm_llama3.1_qlora.yaml) **             | QLoRA  | H100 x 1  | 48 GiB (8.0 GiB)      | 26.1   |
| [Llama 3.1 70B](config/llm_llama3.1_70B_lora_fsdp.yaml) *     | LoRA   | H100 x 2  | 66 GiB (CPU Offload)  | 40.2   |
| [Llama 3 8B](config/llm_llama3_full.yaml)                     | Full   | H100 x 1  | 78 GiB (16 GiB)       | 4.7    |
| [Llama 3 8B](config/llm_llama3_lora.yaml)                     | LoRA   | H100 x 1  | 36 GiB (16 GiB)       | 6.4    |
| [Llama 3 8B](config/llm_llama3_qlora.yaml) **                 | QLoRA  | H100 x 1  | 48 GiB (8.0 GiB)      | 26.1   |
| [Llama 2 13B](config/llm_llama2_full_fsdp.yaml) *             | Full   | H100 x 2  | 31 GiB (CPU Offload)  | 9.5    |  
| [Llama 2 13B](config/llm_llama2_lora.yaml)                    | LoRA   | H100 x 1  | 43 GiB (25.5 GiB)     | 9.8    |
| [Llama 2 13B](config/llm_llama2_qlora.yaml) **                | QLoRA  | H100 x 1  | 38 GiB (8.3 GiB)      | 43.0   |
| [Gemma 2 9B](config/llm_gemma2_full_fsdp.yaml) *              | Full   | H100 x 2  | 59 GiB (CPU Offload)  | 12.6   |
| [Gemma 2 9B](config/llm_gemma2_lora.yaml)                     | LoRA   | H100 x 1  | 60 GiB (18 GiB)       | 12.9   |
| [Gemma 2 9B](config/llm_gemma2_qlora.yaml) **                 | QLoRA  | H100 x 1  | OOM (8.4 GiB)         | OOM    |
| [Gemma 7B](config/llm_gemma_full_fsdp.yaml) *                 | Full   | H100 x 2  | 48 GiB (CPU Offload)  | 8.7    |   
| [Gemma 7B](config/llm_gemma_lora.yaml)                        | LoRA   | H100 x 1  | 51 GiB (17 GiB)       | 9.5    |
| [Gemma 7B](config/llm_gemma_qlora.yaml) **                    | QLoRA  | H100 x 1  | 70 GiB (7.5 GiB)      | 27.4   |
| [Phi3-mini (3.8B)](config/llm_phi3_full.yaml)                 | Full   | H100 x 1  | 40 GiB (8 GiB)        | 4.0    |
| [Phi3-mini (3.8B)](config/llm_phi3_lora.yaml)                 | LoRA   | H100 x 1  | 17 GiB (8 GiB)        | 5.0    |
| [Phi3-mini (3.8B)](config/llm_phi3_qlora.yaml) **             | QLoRA  | H100 x 1  | 21 GiB (3.2 GiB)      | 17.6   |

*: FSDP training with CPU offloading + 32 gradient accumuation.<br>
**: 4-bit QLoRA training. QLoRA does not always use less GPU than LoRA, but it varies depending on sequence length and model size. Experimentally, QLoRA use less GPU when less than 1,500 sequence length. Please refer to [Google document](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/lora-qlora).

&nbsp;

&nbsp;

## Quick Starts 🚀
### Environment Setup
We have to install PyTorch and other requirements. Please refer to more [detailed setup](./docs/1_getting_started.md) including Docker.
```bash
# PyTorch Install
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Requirements Install
pip3 install -r docker/requirements.txt
```

&nbsp;

### Data preparation
```bash
python3 src/run/dataset_download.py --dataset allenai/ai2_arc --download_path data_examples
```

&nbsp;

### LLM Training
```bash
# Llama 3.1 8B LoRA fine-tuning
python3 src/run/train.py --config config/example_llama3.1_lora.yaml --mode train

# Llama 3.1 8B QLoRA fine-tuning
python3 src/run/train.py --config config/example_llama3.1_qlora.yaml --mode train

# Llama 3.1 8B full fine-tuning
python3 src/run/train.py --config config/example_llama3.1_full.yaml --mode train
```

&nbsp;

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
1. [Getting Started](./docs/1_getting_started.md)
2. [Data Preparation](./docs/2_data_preparation.md)
3. [Training](./docs/3_training.md)
