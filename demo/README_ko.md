# LLM Demo

## Introduction 
여기서는 학습된 LLM을 서버에 띄우는 방법과 화면에서 prompting을 할 수 있는 HTML 페이지 코드를 제공합니다.

## Execution Method
### 1. Server Execution
#### 1.1 Arguments
`demo/server.py`를 실행하기 위한 몇 가지 argument가 있습니다.
* [`-m`, `--model_dir`]: 학습된 LLM 모델 디렉토리.
* [`-l`, `--load_model_type`]: [`metric`, `loss`, `last`] 중 하나를 선택.
    * `metric`(default): Valdiation metric (BLEU, ROUGE, etc.)이 최대일 때 모델을 resume.
    * `loss`: Valdiation loss가 최소일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.
* [`-t`, `--template_path`]: 템플릿 json 파일 경로.
* [`-d`, `--device`]: (default: `0`) 사용할 GPU id.
* [`-c`, `--config`]: `config` 폴더의 `config.yaml`. 만약 `--model_dir` 대신 `--config` argument 가 사용되면 pre-trained LLM 모델이 로드 됨.
* [`--is_greedy`]: 사용 시, greedy generation 사용.
* [`--efficient_load`]: GPU 용량이 부족할 시 사용.
* [`--save_context`]: 사용 시, mult-turn chat 사용

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