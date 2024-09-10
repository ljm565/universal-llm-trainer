# Getting Start
In this project, you can build a Docker image or Ananconda environment for an LLM training.

## Docker
### 1. Image Build
You can build a Docker image using the following command:
```bash
cd docker
docker build -t {$IMAGE_NAME} .
```

### 2. Run Docker Container
Then, you can make a Docker container using the following command:
```bash
docker run -it -d --name {$CONTAINER_NAME} --gpus all --shm-size=2g -v {$PATH_TO_BE_MOUNTED}:{$MOUNT_PATH} -v {$PATH_OF_HUGGINGFACE_HUB_CACHE_FOLDER}:/root/.cache/ -v {$PATH_OF_NLTK_DATA_FOLDER}:/root/nltk_data {$IMAGE_NAME}
docker exec -it {$CONTAINER_NAME} /bin/bash
```
<br><br>

## Anaconda
### 0. Preliminary
It is assumed that the Conda environment and Python and PyTorch related libaray are all installed.
PyTorch version 2.0.1 or higher is recommended.
```bash
# torch install example
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1. Package Installation
You can install packages using the following command:
```bash
cd docker
pip3 install -r requirements.txt
```

* If you encounter the bitsandbytes installation error while installing via `requirements.txt`, you can install bitsandbytes via wheel file: 
```bash
pip3 install bitsandbytes-0.42.0-py3-none-any.whl
```