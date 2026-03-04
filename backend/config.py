from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    LEGACY_DIR: Path = Path(__file__).resolve().parent.parent / "processed_pbp"
    WEIGHT_CSV: Path = Path(__file__).resolve().parent.parent / "eFG_Weight.csv"

    # FastAPI
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # NBA API
    NBA_API_DELAY: float = 0.6
    NBA_API_MAX_RETRIES: int = 3

    model_config = {"env_prefix": "NBA_"}


settings = Settings()
