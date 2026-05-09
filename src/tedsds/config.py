from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    pg_dsn: str = "postgresql://tedsds:tedsds@localhost:5432/tedsds"
    data_dir: Path = Path("./data")


settings = Settings()
