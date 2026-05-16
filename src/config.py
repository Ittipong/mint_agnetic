"""Application configuration via Pydantic Settings."""

from typing import Literal

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

    # Intent classifier — runs on every text-only turn to decide whether
    # the user is asking to RECORD a new transaction ("กิน kfc 100บาท")
    # vs anything else (analytics question, chit-chat). The output
    # routes between the quick-add lane and the regular ReAct lane.
    # Picked for low cost + low latency; OpenRouter exposes Gemini.
    intent_classifier_model: str = "google/gemini-2.5-flash-lite"
    intent_classifier_base_url: str = "https://openrouter.ai/api/v1"

    # Speech-to-text — used by the /chat/voice multipart endpoint to
    # transcribe an uploaded audio clip (m4a/aac/mp3/wav, ≤60s) before
    # the rest of the graph runs. Gemini 2.0 Flash Lite is the cheapest
    # multimodal model on OpenRouter that accepts audio_inputs.
    stt_model: str = "google/gemini-2.0-flash-lite-001"
    stt_base_url: str = "https://openrouter.ai/api/v1"

    # Deployment environment — controls Postel's Law fallbacks per the
    # project rule: in production, sync/voice endpoints tolerate
    # imperfect payloads (missing optional fields, oddly-named mime
    # types) and log warnings; in dev/test the same code paths must
    # fail loudly so bad data is surfaced before it ships.
    environment: Literal["development", "staging", "test", "production"] = "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # `debug` writes every node-internal detail (state_keys, catalog
    # counts, history compression, etc.) — useful while iterating on
    # prompts/tools. `production` drops to milestones only (HTTP entry,
    # STREAM start/done/error, tool start/end + duration, LLM tool_calls
    # summary). Toggle via env var `LOG_MODE`.
    log_mode: Literal["production", "debug"] = "debug"

    @property
    def is_debug_log(self) -> bool:
        return self.log_mode == "debug"


settings = Settings()
