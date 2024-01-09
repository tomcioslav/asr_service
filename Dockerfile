FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies, including Python, Git, ZSH, and wget
RUN apt-get update \
    && apt-get install -y software-properties-common git zsh wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py

# Set Python3.11 as the default python
RUN PIP_EXECUTABLE=$(which pip) && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip $PIP_EXECUTABLE 1

# Install Poetry
RUN pip install --no-cache-dir poetry==1.4.2

# Set the working directory in the Docker image
WORKDIR /asr_service

# Copy the pyproject.toml and other important files into the image
COPY README.md pyproject.toml main.py /asr_service/
COPY models/  /asr_service/models/
COPY src/asr_service/  /asr_service/src/asr_service/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

