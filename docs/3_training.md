# Training
This document provides a guide for instruction-tuning (or fine-tuning) LLMs.

## Configuration Preparation
To train an LLM model, you need to create a configuration. Detailed explanations and examples of the options for the configuration are as follows.
<details>
<summary>Configuration example</summary>

```yaml
# base
train_type: llm
seed: 0
deterministic: True
model: gemma
model_size: 2b     # valid pattern examples: 10b, 1.2b, 2.8, 5 (you can omit b character)

# data config
data_path: ['data/mras/mras_en_otc_v4', 'data/mras/mras_ko_rpr_v4']
template_dir: ['templates/gemma_templates', 'templates/gemma_templates_ko']   # list(path) or path, if null, template will be chosen according to the ${data_path}/templates

# project config
project: outputs/gemma
name: mras_v4

# environment config
device: [0,1]
bit: 8                          # [4, 8, 16, 32], If 4, 8, model will be quantized
load_unnecessary_half: False    # recommend to set True if you set bit to 16 (4, 8bit model automatically loads neccesaries to 32bit)
half_inference: False

# data config
max_length: 2000                             # max sequnece length
is_multi_turn: False                         # multi-turn training option
add_bos_token_when_response_start: True      # if True and bos_token is existing, bos token will be added at the first of the input
add_eos_token_when_response_end: True        # if False and eos_token is existing, eos token will be added at the last of the input
data_verbose: True                           # if True, data statistics will be calculated and graphs are saved in the project folder

# tokenizer config (I recommend to double check the tokenizer's special token map)
pad_token_id: null    # [add, null, int] if null, tokenizer pad_token_id will not be overrided
bos_token_id: null    # [add, null, int] if null, tokenizer bos_token_id will not be overrided
eos_token_id: null    # [add, null, int] if null, tokenizer eos_token_id will not be overrided
cls_token_id: null    # [add, null, int] if null, tokenizer cls_token_id will not be overrided
sep_token_id: null    # [add, null, int] if null, tokenizer sep_token_id will not be overrided
unk_token_id: null    # [add, null, int] if null, tokenizer unk_token_id will not be overrided

# training config
batch_size: 2
epochs: 300           # if optimizer_step_criterion set to 'epoch' it will be activated and scheduler will step per epoch
warmup_epochs: 0      # if optimizer_step_criterion set to 'epoch' it will be activated
steps: 300000         # if optimizer_step_criterion set to 'step' it will be activated and scheduler will step per step
warmup_steps: 100     # if optimizer_step_criterion set to 'step' it will be activated
optimizer_step_criterion: 'step'   # ['epoch', 'step']
lr0: 1e-4
lrf: 0.001                 # last_lr = lr0 * lrf
scheduler_type: 'cosine'   # ['linear', 'cosine']
momentum: 0.9
weight_decay: 0.0
warmup_momentum: 0.8
early_stop_criterion: 50
workers: 0
amp_training: False
ema_updating: False
train_verbose: True
inference_result_verbose: True      # If True, inference results will be printed at the validation phase

# peft
peft_config_path: config/gemma_lora_config.yaml       # if False, training without peft. Also, if you use LoRA with quantization, it would be QLoRA

# logging data and metrics
common: ['train_loss', 'validation_loss', 'lr']               # ['train_loss', 'validation_loss', 'lr']
metrics: ['ppl', 'bleu', 'rouge', 'edit_distance', 'meteor']  # ['ppl', 'bleu', 'edit_distance', 'rouge', 'meteor']  # TODO: CIDEr 추가
fast_validation_n: null                                       # [null, 1, 2, ...], Only the number of data of the set value is evaluated per step
fast_validation_step_interval: null                           # [null, 1, 2, ...], validation step inteval. if null, all validation steps will be executed.
validation_step_interval_prop: 0.3                            # setting between 0 and 1 values
tensorboard_logging_interval: 1                               # tensorboard logging step
```
</details>
<br><br>

## Training Strategy
### 1. Distributed Data Paralle (DDP)
This repository supports DDP training. You can simply set up multiple devices at the "device" part as shown in the above  of `config` example.
```yaml
device: [0,1,2]  # This will be use Rank 0, 1 GPUs
```
```yaml
# In this case, CUDA_VISIBLE_DEVICES are automatically set to 0, 1, 2.
device: [3,4,5]  
```

### 2. Fully Sharded DataParallel (FSDP)
TBA
<br><br><br>


## Instruction & Fine-tuning
### 1. Data Preparation
Before training the model, please refer to [Data Preparation](./2_data_preparation.md) to prepare the data.

### 2. Training
#### Examples of training commands:
```bash
# For more option, please refer to train.py
# Training from scratch
python3 src/run/train.py --config config/llm_qwen.yaml --mode train

# Training resume
python3 src/run/train.py --config config/llm_qwen.yaml --mode resume --resume_model_dir outputs/chat/qwen_fine_tuned --load_model_type best

# deepspeed training
python3 src/run/trian_deepspeed.py --config config/llm_llama3.yaml --mode train
```

#### Examples of trained model inferencing commands: 
```bash
# For more option, please refer to validation.py
# Validation vanilla pre-trained model
python3 src/run/valiation.py --config config/llm_qwen.yaml

# Validation our instruction-tuned model
python3 src/run/validation.py --resume_model_dir config/llm_qwen.yaml --load_model_type best
```
<br><br>


## Adding New Datasets and Models
### 1. Adding a New Model
To test a new LLM, you only need to create a few wrappers as follows:
1. Create a new wrapper in `src/models/${new_model}.py`.
2. Create a new tokenizer wrapper in `src/tools/tokenizers/${new_model_tokenizer}.py`.
3. Add it to the `get_model` function in `src/trainer/build.py`.
4. Add it to the `choose_proper_model` function in `src/utils/training_utils.py`.

### 2. Adding a New Dataset
To train with a new dataset, you need to add it to the following function:
1. Add it to the `choose_proper_dataset` function in `src/utils/data_utils.py`.
