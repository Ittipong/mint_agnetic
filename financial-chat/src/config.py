"""Application configuration via Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment variables for mint-agentic."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str
    database_url: str = "postgresql://mint:mint123@localhost:5433/mint_agentic"
    backend_database_url: str = "postgresql://postgres:postgres@localhost:5432/mint_money_dev"
    langgraph_studio_port: int = 8080

    model: str = "google/gemma-4-26b-a4b-it"
    base_url: str = "https://openrouter.ai/api/v1"
    intent_model: str = "mistralai/ministral-3b-2512"

    server_host: str = "0.0.0.0"
    server_port: int = 8000


settings = Settings()
