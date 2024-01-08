FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Poetry
RUN pip install --no-cache-dir poetry==1.4.2

# Set the working directory in the Docker image
WORKDIR /asr_service

# Copy the pyproject.toml and other important files into the image
COPY README.md pyproject.toml  /asr_service/
COPY models/  /asr_service/models/
COPY src/asr_service/  /asr_service/src/asr_service/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

