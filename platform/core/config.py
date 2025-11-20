"""Application configuration using pydantic-settings."""
from functools import lru_cache
from typing import Optional

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Database
    database_url: PostgresDsn
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Redis
    redis_url: RedisDsn

    # GCP
    gcp_project_id: str
    gcs_bucket_artifacts: str
    gcs_bucket_datasets: str

    # API Security
    api_secret_key: str
    api_algorithm: str = "HS256"
    api_access_token_expire_minutes: int = 15
    api_refresh_token_expire_days: int = 7

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    # Celery
    celery_broker_url: str
    celery_result_backend: str

    # W&B
    wandb_project_prefix: str = "llm-experiments"

    # Feature Flags
    enable_anthropic: bool = True
    enable_custom_evaluators: bool = True

    # Environment
    environment: str = "development"
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
