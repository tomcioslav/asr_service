from pathlib import Path

from pydantic_settings import BaseSettings

class Paths(BaseSettings):
    BASE: Path = Path(__file__).parent.parent.parent
    DATA: Path = BASE / "data"

paths = Paths()

class Credentials(BaseSettings):
    SAMPLE_PASS: str
    
    class Config:
        env_file = paths.BASE / ".env"
        env_file_encoding = "utf-8"

credentials = Credentials()