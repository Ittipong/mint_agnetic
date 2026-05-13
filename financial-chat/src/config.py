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
    react_model: str = ""
    react_base_url: str = ""
    # Fallback models for ReAct — only active when react_base_url is OpenRouter.
    react_fallback_models: list[str] = Field(
        default=[],
    )

    # CodeAct agent model (OpenRouter)
    codeact_model: str = ""
    codeact_base_url: str = ""
    # Fallback models tried in order when primary is down/rate-limited.
    codeact_fallback_models: list[str] = Field(
        default=[],
    )

    # Vision model — used by the slip-to-transaction subgraph only.
    # Must be a multimodal model that accepts `image_url` content blocks
    # in OpenAI-compatible chat format. Failure to invoke surfaces as
    # an SSE error to the mobile client (no silent fallback).
    vision_model: str = "bytedance-seed/seedream-4.5"
    vision_base_url: str = "https://openrouter.ai/api/v1"

    server_host: str = "0.0.0.0"
    server_port: int = 8000


settings = Settings()
