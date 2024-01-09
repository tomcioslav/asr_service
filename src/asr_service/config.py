from pathlib import Path
import torch

from pydantic_settings import BaseSettings

from asr_service.schema import ModelSize


class Paths(BaseSettings):
    BASE: Path = Path(__file__).parent.parent.parent
    DATA: Path = BASE / "data"
    DATA_LOCAL: Path = DATA / "local"
    MODEL_LARGE: Path = BASE / "models" / "models--openai--whisper-large-v3"/"snapshots"/"bf128ed72b9ea8cd29be04376f1dd1b9d418a2a5"
    MODEL_MEDIUM: Path = BASE / "models" / "models--openai--whisper-medium"/"snapshots"/"353117b351a2a3d740c3bdbba1396b06e2499bde"
    MODEL_SMALL: Path = BASE / "models" / "models--openai--whisper-small"/"snapshots"/"ee34e8ae444c29815eca53e11383ea13b2e362eb0"
paths = Paths()

class ModelParams(BaseSettings):
    MODEL_SIZE: ModelSize = "small"
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
model_params = ModelParams()

