from pathlib import Path
import torch
import os

from pydantic_settings import BaseSettings

from asr_service.schema import ModelSize


class Paths(BaseSettings):
    BASE: Path = Path(__file__).parent.parent.parent
    MODELS: Path = BASE / "models"
    MODEL_LARGE: Path = MODELS / "whisper-large-v3"
    MODEL_MEDIUM: Path = MODELS / "whisper-medium"
    MODEL_SMALL: Path = MODELS / "whisper-small"
    DATA: Path = BASE / "data"
paths = Paths()

# Ensure models directory exists
paths.MODELS.mkdir(exist_ok=True)

# Set Hugging Face cache directories
os.environ["HF_HOME"] = str(paths.MODELS)
os.environ["TRANSFORMERS_CACHE"] = str(paths.MODELS / "transformers")

class ModelParams(BaseSettings):
    MODEL_SIZE: ModelSize = "small"
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
model_params = ModelParams()

