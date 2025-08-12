# GPU Test
Here, we present the results of testing LLM training on multiple GPUs using this repository.
Compared to training under the same conditions as torchtune, we observed advantages in both speed and GPU memory usage.

&nbsp;

## GPU memory and training speed
Below is an example of the memory requirements and training speed for different models.

### 1. NVIDIA H100
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
| [Llama 3.1 8B](../config//llm_llama3.1_full.yaml)                 | Full   | H100 x 1  | 78 GiB (16 GiB)       | 4.7    |
| [Llama 3.1 8B](../config//llm_llama3.1_lora.yaml)                 | LoRA   | H100 x 1  | 36 GiB (16 GiB)       | 6.4    |
| [Llama 3.1 8B](../config//llm_llama3.1_qlora.yaml) **             | QLoRA  | H100 x 1  | 48 GiB (8.0 GiB)      | 26.1   |
| [Llama 3.1 70B](../config//llm_llama3.1_70B_lora_fsdp.yaml) *     | LoRA   | H100 x 2  | 66 GiB (CPU Offload)  | 40.2   |
| [Llama 3 8B](../config//llm_llama3_full.yaml)                     | Full   | H100 x 1  | 78 GiB (16 GiB)       | 4.7    |
| [Llama 3 8B](../config//llm_llama3_lora.yaml)                     | LoRA   | H100 x 1  | 36 GiB (16 GiB)       | 6.4    |
| [Llama 3 8B](../config//llm_llama3_qlora.yaml) **                 | QLoRA  | H100 x 1  | 48 GiB (8.0 GiB)      | 26.1   |
| [Llama 2 13B](../config//llm_llama2_full_fsdp.yaml) *             | Full   | H100 x 2  | 31 GiB (CPU Offload)  | 9.5    |  
| [Llama 2 13B](../config//llm_llama2_lora.yaml)                    | LoRA   | H100 x 1  | 43 GiB (25.5 GiB)     | 9.8    |
| [Llama 2 13B](../config//llm_llama2_qlora.yaml) **                | QLoRA  | H100 x 1  | 38 GiB (8.3 GiB)      | 43.0   |
| [Gemma 2 9B](../config//llm_gemma2_full_fsdp.yaml) *              | Full   | H100 x 2  | 59 GiB (CPU Offload)  | 12.6   |
| [Gemma 2 9B](../config//llm_gemma2_lora.yaml)                     | LoRA   | H100 x 1  | 60 GiB (18 GiB)       | 12.9   |
| [Gemma 2 9B](../config//llm_gemma2_qlora.yaml) **                 | QLoRA  | H100 x 1  | OOM (8.4 GiB)         | OOM    |
| [Gemma 7B](../config//llm_gemma_full_fsdp.yaml) *                 | Full   | H100 x 2  | 48 GiB (CPU Offload)  | 8.7    |   
| [Gemma 7B](../config//llm_gemma_lora.yaml)                        | LoRA   | H100 x 1  | 51 GiB (17 GiB)       | 9.5    |
| [Gemma 7B](../config//llm_gemma_qlora.yaml) **                    | QLoRA  | H100 x 1  | 70 GiB (7.5 GiB)      | 27.4   |
| [Phi3-mini (3.8B)](../config//llm_phi3_full.yaml)                 | Full   | H100 x 1  | 40 GiB (8 GiB)        | 4.0    |
| [Phi3-mini (3.8B)](../config//llm_phi3_lora.yaml)                 | LoRA   | H100 x 1  | 17 GiB (8 GiB)        | 5.0    |
| [Phi3-mini (3.8B)](../config//llm_phi3_qlora.yaml) **             | QLoRA  | H100 x 1  | 21 GiB (3.2 GiB)      | 17.6   |

*: FSDP training with CPU offloading + 32 gradient accumuation.<br>
**: 4-bit QLoRA training. QLoRA does not always use less GPU than LoRA, but it varies depending on sequence length and model size. Experimentally, QLoRA use less GPU when less than 1,500 sequence length. Please refer to [Google document](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/lora-qlora).

&nbsp;

### 2. NVIDIA RTX Pro 6000
> [!NOTE]
> Training conditions: 
> - Environments: Ubuntu 22.04.5 LTS, torch==2.8.0, transformers==4.49.0, bitsandbytes==0.46.1
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
| [Llama 3.1 8B](../config/llm_llama3.1_full.yaml)                 | Full   | Pro 6000 x 1  | 89 GiB (16 GiB)       | 10.9    |
| [Llama 3.1 8B](../config/llm_llama3.1_lora.yaml)                 | LoRA   | Pro 6000 x 1  | 37 GiB (16 GiB)       | 12.5    |
| [Llama 3.1 8B](../config/llm_llama3.1_qlora.yaml) **             | QLoRA  | Pro 6000 x 1  | 48 GiB (8.0 GiB)      | 30.7    |
| [Llama 3.1 70B](../config/llm_llama3.1_70B_lora_fsdp.yaml) *     | LoRA   | Pro 6000 x 2  | 65 GiB (CPU Offload)  | 82.7    |
| [Llama 3 8B](../config/llm_llama3_full.yaml)                     | Full   | Pro 6000 x 1  | 89 GiB (16 GiB)       | 10.9    |
| [Llama 3 8B](../config/llm_llama3_lora.yaml)                     | LoRA   | Pro 6000 x 1  | 37 GiB (16 GiB)       | 12.5    |
| [Llama 3 8B](../config/llm_llama3_qlora.yaml) **                 | QLoRA  | Pro 6000 x 1  | 48 GiB (8.0 GiB)      | 30.7    |
| [Llama 2 13B](../config/llm_llama2_full_fsdp.yaml) *             | Full   | Pro 6000 x 2  | 31 GiB (CPU Offload)  | 22.3    |  
| [Llama 2 13B](../config/llm_llama2_lora.yaml)                    | LoRA   | Pro 6000 x 1  | 43 GiB (26.2 GiB)     | 19.6    |
| [Llama 2 13B](../config/llm_llama2_qlora.yaml) **                | QLoRA  | Pro 6000 x 1  | 38 GiB (8.0 GiB)      | 52.0    |
| [Gemma 2 9B](../config/llm_gemma2_full_fsdp.yaml) *              | Full   | Pro 6000 x 2  | 57 GiB (CPU Offload)  | 18.9    |
| [Gemma 2 9B](../config/llm_gemma2_lora.yaml)                     | LoRA   | Pro 6000 x 1  | 60 GiB (18.5 GiB)     | 17.2    |
| [Gemma 2 9B](../config/llm_gemma2_qlora.yaml) **                 | QLoRA  | Pro 6000 x 1  | 89 GiB (8.2 GiB)      | 40.3    |
| [Gemma 7B](../config/llm_gemma_full_fsdp.yaml) *                 | Full   | Pro 6000 x 2  | 47 GiB (CPU Offload)  | 13.5    |   
| [Gemma 7B](../config/llm_gemma_lora.yaml)                        | LoRA   | Pro 6000 x 1  | 51 GiB (17 GiB)       | 13.1    |
| [Gemma 7B](../config/llm_gemma_qlora.yaml) **                    | QLoRA  | Pro 6000 x 1  | 70 GiB (7.2 GiB)      | 33.2    |
| [Phi3-mini (3.8B)](../config/llm_phi3_full.yaml)                 | Full   | Pro 6000 x 1  | 40 GiB (7.6 GiB)	    | 8.0     |
| [Phi3-mini (3.8B)](../config/llm_phi3_lora.yaml)                 | LoRA   | Pro 6000 x 1  | 17 GiB (7.6 GiB)      | 9.2     |
| [Phi3-mini (3.8B)](../config/llm_phi3_qlora.yaml) **             | QLoRA  | Pro 6000 x 1  | 21 GiB (2.7 GiB)      | 20.7    |

*: FSDP training with CPU offloading + 32 gradient accumuation.<br>
**: 4-bit QLoRA training. QLoRA does not always use less GPU than LoRA, but it varies depending on sequence length and model size. Experimentally, QLoRA use less GPU when less than 1,500 sequence length. Please refer to [Google document](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/lora-qlora).

&nbsp;

### 3. NVIDIA A100
> [!NOTE]
> Training conditions: 
> - Environments: Ubuntu 22.04.6 LTS, torch==2.5.1, transformers==4.49.0
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
| [Llama 3.1 8B](../config/llm_llama3.1_full.yaml)                 | Full   | A100 x 1  | 78 GiB (16 GiB)           |  9.7    |
| [Llama 3.1 8B](../config/llm_llama3.1_lora.yaml)                 | LoRA   | A100 x 1  | 36 GiB (16 GiB)           |  11.5   |
| [Llama 3.1 8B](../config/llm_llama3.1_qlora.yaml) **             | QLoRA  | A100 x 1  | 48 GiB (7.8 GiB)          |  57.6   |
| [Llama 3.1 70B](../config/llm_llama3.1_70B_lora_fsdp.yaml) *     | LoRA   | A100 x 2  | 65 GiB (CPU Offload)      |  84.1   |
| [Llama 3 8B](../config/llm_llama3_full.yaml)                     | Full   | A100 x 1  | 78 GiB (16 GiB)           |  9.7    |
| [Llama 3 8B](../config/llm_llama3_lora.yaml)                     | LoRA   | A100 x 1  | 36 GiB (16 GiB)           |  11.5   |
| [Llama 3 8B](../config/llm_llama3_qlora.yaml) **                 | QLoRA  | A100 x 1  | 48 GiB (7.8 GiB)          |  57.6   |
| [Llama 2 13B](../config/llm_llama2_full_fsdp.yaml) *             | Full   | A100 x 2  | 31 GiB (CPU Offload)      | 25.4    |  
| [Llama 2 13B](../config/llm_llama2_lora.yaml)                    | LoRA   | A100 x 1  | 43 GiB (26.2 GiB)         | 17.9    |
| [Llama 2 13B](../config/llm_llama2_qlora.yaml) **                | QLoRA  | A100 x 1  | 38 GiB (8.0 GiB)          | 96.5    |
| [Gemma 2 9B](../config/llm_gemma2_full_fsdp.yaml) *              | Full   | A100 x 2  | 58 GiB (CPU Offload)      | 25.2    |
| [Gemma 2 9B](../config/llm_gemma2_lora.yaml)                     | LoRA   | A100 x 1  | 60 GiB (18.5 GiB)         | 20.0    |
| [Gemma 2 9B](../config/llm_gemma2_qlora.yaml) **                 | QLoRA  | A100 x 1  | OOM (8.2 GiB)             | OOM     |
| [Gemma 7B](../config/llm_gemma_full_fsdp.yaml) *                 | Full   | A100 x 2  | 48 GiB (CPU Offload)      | 18.5    |   
| [Gemma 7B](../config/llm_gemma_lora.yaml)                        | LoRA   | A100 x 1  | 51 GiB (17 GiB)           | 14.7    |
| [Gemma 7B](../config/llm_gemma_qlora.yaml) **                    | QLoRA  | A100 x 1  | 70 GiB (7.2 GiB)          | 61.5    |
| [Phi3-mini (3.8B)](../config/llm_phi3_full.yaml)                 | Full   | A100 x 1  | 40 GiB (7.7 GiB)          | 7.4     |
| [Phi3-mini (3.8B)](../config/llm_phi3_lora.yaml)                 | LoRA   | A100 x 1  | 17 GiB (7.6 GiB)          | 8.5     |
| [Phi3-mini (3.8B)](../config/llm_phi3_qlora.yaml) **             | QLoRA  | A100 x 1  | 21 GiB (2.7 GiB)          | 36.1    |

*: FSDP training with CPU offloading + 32 gradient accumuation.<br>
**: 4-bit QLoRA training. QLoRA does not always use less GPU than LoRA, but it varies depending on sequence length and model size. Experimentally, QLoRA use less GPU when less than 1,500 sequence length. Please refer to [Google document](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/lora-qlora).

&nbsp;