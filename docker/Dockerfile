# # Use the official PyTorch 2.5.1 image
# FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Use the official PyTorch 2.2.2 image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /workspace/llm

# Install C extensions, git, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    vim \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install packages
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

# # Install bitsandbytes via wheel file if you failed to install pip install bitsandbytes
# COPY bitsandbytes-0.42.0-py3-none-any.whl .
# RUN pip install bitsandbytes-0.42.0-py3-none-any.whl
# RUN rm bitsandbytes-0.42.0-py3-none-any.whl


# Copy your PyTorch application files into the container
# COPY . /app

# Set the default command to run when the container starts
# CMD ["python", "your_script.py"]
