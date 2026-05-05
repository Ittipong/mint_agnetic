"""Application configuration via Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Environment variables for mint-agentic."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = ""
    typhoon_api_key: str = Field(default="", validation_alias="TYPOHOON_API_KEY")
    database_url: str = "postgresql://mint:mint123@localhost:5433/mint_agentic"
    backend_database_url: str = "postgresql://postgres:postgres@localhost:5432/mint_money_dev"
    langgraph_studio_port: int = 8080

    # ReAct agent model (Typhoon)
    react_model: str = "typhoon-v2.5-30b-a3b-instruct"
    react_base_url: str = "https://api.opentyphoon.ai/v1"

    # CodeAct agent model (OpenRouter)
    codeact_model: str = "openai/gpt-oss-safeguard-20b:nitro"
    codeact_base_url: str = "https://openrouter.ai/api/v1"

    # Smart CodeAct — when True, every analyze_user_finances call is routed
    # through the CodeAct loop (with resolve_*/parse_period helpers) instead
    # of the legacy plan→entity_resolve→time_resolve→execute pipeline.
    smart_codeact_enabled: bool = False

    server_host: str = "0.0.0.0"
    server_port: int = 8000


settings = Settings()
