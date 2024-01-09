from pathlib import Path

from pydantic_settings import BaseSettings


class Paths(BaseSettings):
    BASE: Path = Path(__file__).parent.parent.parent
    DATA: Path = BASE / "data"
    DATA_LOCAL: Path = DATA / "local"
    WHISPER: Path = BASE / "models" / "models--openai--whisper-large-v3"/"snapshots"/"bf128ed72b9ea8cd29be04376f1dd1b9d418a2a5"


paths = Paths()


class Credentials(BaseSettings):
    class Config:
        env_file = paths.BASE / ".env"
        env_file_encoding = "utf-8"


credentials = Credentials()
