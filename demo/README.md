# LLM Demo
한국어 버전의 설명은 [여기](./README_ko.md)를 참고하시기 바랍니다.

## Introduction 
Here, we provide the method to deploy a trained LLM on a server and the code for an HTML page that allows you to perform prompting on the screen.


## Execution Method
### 1. Server Execution
#### 1.1 Arguments
There are several arguments for running `demo/server.py`:
* [`-m`, `--model_dir`]: Trained LLM model directory.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's metrics such as BLEU, ROUGE, etc.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-t`, `--template_path`]: Template json file path.
* [`-d`, `--device`]: (default: `0`) GPU number to be executed.
* [`-c`, `--config`]: Path of `config.yaml` in the `config` folder. If `--config` argument is provided without a `--model_dir`, a pre-trained LLM will be loaded.
* [`--is_greedy`]: If it activated, greedy generation will be activated.
* [`--efficient_load`]: When there is insufficient GPU space available, please use itactivated.
* [`--save_context`]: If it activated, multi-turn chat will be available.

#### 2.2 Command
```bash
# execute LLM
python3 demo/server.py --model_dir ${project}/${name} --template_path ${TEMPLATE_JSON_PATH}

# execute LLM with greedy generation
python3 demo/server.py --model_dir ${project}/${name} --template_path ${TEMPLATE_JSON_PATH} --is_greedy

# execute just pre-training LLM (not your customed LLM)
python3 demo/server.py --config config/${CONFIT_YAML}
```
<br><br>

### 2. Prompting Page Execution
#### 2.1. Move to front folder
```bash
cd demo/front
```

#### 2.2. Execute the demo page
```bash
# if PORT is 8888, you can see the demo page at "localhost:8888" 
python3 -m http.server ${PORT}
```
<br>
