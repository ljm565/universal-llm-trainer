# Getting Start
본 프로젝트에서는 anaconda 혹은 docker container를 이용하여 환경을 구축할 수 있습니다.

## Docker
### 1. Image Build
다음과 같은 명령어로 docker image를 build 합니다.
```bash
cd docker
docker build -t {$IMAGE_NAME} .
```

### 2. Run Docker Container
다음과 같은 명령어로 docker container를 run 합니다.
```bash
docker run -it -d --name {$CONTAINER_NAME} --gpus all --shm-size=2g -v {$PATH_TO_BE_MOUNTED}:{$MOUNT_PATH} -v {$PATH_OF_HUGGINGFACE_HUB_CACHE_FOLDER}:/root/.cache/ -v {$PATH_OF_NLTK_DATA_FOLDER}:/root/nltk_data {$IMAGE_NAME}
docker exec -it {$CONTAINER_NAME} /bin/bash
```
<br><br>

## Anaconda
### 0. Preliminary
Conda 환경과 Python, PyTorch 관련 libaray가 모두 설치가 되어있음을 가정합니다.
PyTorch는 2.0.1 버전 이상을 권장합니다. 
```bash
# torch install example
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1. Package Installation
다음과 같은 명령어로 pacakge를 설치합니다.
```bash
cd docker
pip3 install -r requirements.txt
```

* 만약 `requirements.txt`의 bitsandbytes 설치 오류가 날 경우 wheel 파일로 설치합니다.
```bash
pip3 install bitsandbytes-0.42.0-py3-none-any.whl
```