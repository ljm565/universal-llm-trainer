# Training
This document provides a guide for instruction-tuning (or fine-tuning) LLMs.

&nbsp;

## Configuration Preparation
To train an LLM model, you need to create a configuration. Detailed explanations and examples of the options for the configuration are as follows.
<details>
<summary>Configuration example</summary>

```yaml
# base
seed: 0
deterministic: True
model: llama3.1
model_size: 8b     # valid pattern examples: 10b, 1.2b, 2.8, 5 (you can omit b character)

# data config
data_train_type: ['qa']
data_path: ['/llm_data/training_dataset/incorporated_ner_data_v1']
template_dir: [templates/llama3_templates]   # list(directory) or directory, if null, template will be chosen according to the {data_path}/templates.

# project config
project: /full_training/output/llama3.1
name: full_test

# environment config
device: [0]
bit: bfloat16                 # [4, 8, 16, 32] or string like "float16" or "bfloat16", etc.
quant_config: False    # str, if you set bit as 4 or 8, you have to insert appropriate quantization config, or assertion error will occur. It only activate when you set bit to 4 or 8.
attn_implementation: null      # [flash_attention_2, sdpa, eager], if null, None default value will be applied.
half_inference: False

# FSDP training configs
fsdp_train: False
fsdp_hyperparameters:
    cpu_offload: True
    amp_training: True
    wrap_policy: size_based     # [size_based, transformer_based]
    size_based:
        min_num_params: 10000     # integer value
    transformer_based:
        fsdp_layer_cls: null    # List of modules to apply FSDP

# data config
max_length: 8192
is_multi_turn: False
add_bos_token_when_response_start: True
add_eos_token_when_response_end: True
data_verbose: False

# tokenizer config (I recommend to double check the tokenizer's special token map)
pad_token_id: 128002     # [add, null, int] if null, tokenizer pad_token_id will not be overrided
bos_token_id: null    # [add, null, int] if null, tokenizer bos_token_id will not be overrided
eos_token_id: null    # [add, null, int] if null, tokenizer eos_token_id will not be overrided
cls_token_id: null    # [add, null, int] if null, tokenizer cls_token_id will not be overrided
sep_token_id: null    # [add, null, int] if null, tokenizer sep_token_id will not be overrided
unk_token_id: null    # [add, null, int] if null, tokenizer unk_token_id will not be overrided

# training config
batch_size: 2
epochs: 300
warmup_epochs: 0
steps: 284000
warmup_steps: 2000
optimizer_step_criterion: 'step'   # ['epoch', 'step']
lr0: 5e-5
lrf: 0.01      # last_lr = lr0 * lrf
scheduler_type: 'cosine'   # ['linear', 'cosine']
momentum: 0.9
weight_decay: 0.0
warmup_momentum: 0.8
early_stop_criterion: 5
workers: 1
total_cpu_use: 32
amp_training: False
ema_updating: False
train_verbose: True
inference_result_verbose: True
gradient_checkpointing:
    activate: True        # if True, gradient checkpointing strategy will be used to save GPU memeory consumption.
    checkpoint_type: 'hf_checkpoint'     # ['hf_checkpoint', 'torch_checkpoint'], if null, default value (hf_checkpoint) will be applied.
gradient_accumuate_step: 1
generation_max_time: 200        # Maximum time (seconds) of model's generation function. If null, no time limits. 
del_logits_after_forward: True  # If True, you can save GPU memroy consumption.

# peft
peft_config_path: False     # if False, training without peft
adapter_save_type: merge             # ['merge', 'adapter_only']

# logging data and metrics
common: ['train_loss', 'validation_loss', 'lr']               # ['train_loss', 'validation_loss', 'lr']
metrics: ['ppl', 'bleu', 'edit_distance', 'rouge', 'meteor']  # ['ppl', 'bleu', 'edit_distance', 'rouge', 'meteor']  # TODO: CIDEr 추가
fast_validation_n: null                                          # [null, 1, 2, ...], Only the number of data of the set value is evaluated per step
fast_validation_step_interval: 5                            # [null, 1, 2, ...], if null, all validation steps will be executed
validation_step_interval_prop: 1                          # setting between 0 and 1 values
tensorboard_logging_interval: 1
```
</details>

&nbsp;

&nbsp;

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

&nbsp;


### 2. Fully Sharded DataParallel (FSDP)
You can train FSDP when you triggered `fsdp_train` to True with mutiple GPUs setting as same as DDP examples.


&nbsp;

&nbsp;


## Instruction & Fine-tuning
### 1. Data Preparation
Before training the model, please refer to [Data Preparation](./2_data_preparation.md) to prepare the data.

&nbsp;

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

&nbsp;

&nbsp;


## Adding New Datasets and Models
### 1. Adding a New Model
To test a new LLM, you only need to create a few wrappers as follows:
1. Create a new wrapper in `src/models/${new_model}.py`.
2. Create a new tokenizer wrapper in `src/tools/tokenizers/${new_model_tokenizer}.py`.
3. Add it to the `get_model` function in `src/trainer/build.py`.
4. Add it to the `choose_proper_model` function in `src/utils/training_utils.py`.

&nbsp;

### 2. Adding a New Dataset
To train with a new dataset, you need to add it to the following function:
1. Add it to the `choose_proper_dataset` function in `src/utils/data_utils.py`.
