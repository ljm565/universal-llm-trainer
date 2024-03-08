# Getting Start

VELKOZ에서는 conda 혹은 docker container를 이용하여 환경을 구축할 수 있습니다.

## Docker

### 1. Image Build

다음과 같은 명령어로 docker image를 build 합니다.

```bash
cd docker
docker build -t {$IMAGE_NAME} .
```
그리고 다음과 같은 명령어로 서버로 이미지를 올릴 수 있습니다.
```bash
docker save {$IMAGE_NAME} > {$IMAGE_NAME}.tar
scp {$IMAGE_NAME}.tar {$USER}@{$ADDRESS}:{$SAVE_PATH}

# 서버로 이동
cd {$SAVE_PATH}
docker load < {$IMAGE_NAME}.tar 
```

### 2. Run Docker Container

다음과 같은 명령어로 docker container를 run 합니다.

```bash
docker run -it -d --name {$CONTAINER_NAME} --gpus all --shm-size=2g -v {$PATH_TO_BE_MOUNTED}:{$MOUNT_PATH} {$IMAGE_NAME}
docker exec -it {$CONTAINER_NAME} /bin/bash
```


## Anaconda

### 0. Preliminary

Conda 환경과 Python, PyTorch 관련 libaray가 모두 설치가 되어있음을 가정합니다.
PyTorch는 1.13 버전 이상을 권장합니다. 

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