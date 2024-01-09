asr_service
==============================

A simple API to run ASR on audio files. You can specify the model size that you want to use. The API will use the models in the models folder based on paths specified in the src/asr_service/config.py file.

### Getting started
Just run docker-compose up and the API will be available at localhost:8000.

It can use different sizes of Asr engines. The default is the smallest one. To use a different one, just change environment variable in docker-compose.yaml. The available sizes are: small, medium, large.
The API Will download the models from the internet if they are not available in the paths specified in the config file.
