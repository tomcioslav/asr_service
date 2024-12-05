FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies, including Python, Git, ZSH, and wget
RUN apt-get update \
    && apt-get install -y software-properties-common git zsh wget curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip and upgrade it
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3.11 -m pip install --upgrade pip

# Set Python3.11 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 1

RUN apt-get update && apt-get install -y ffmpeg

# Install Poetry with pip and configure it
RUN pip install --no-cache-dir poetry==1.4.2 

# Set the working directory in the Docker image
WORKDIR /asr_service

# Copy working directory contents
COPY . /asr_service/

# Install dependencies using Poetry
RUN poetry config virtualenvs.in-project true && poetry install


