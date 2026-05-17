"""Application configuration via Pydantic Settings.

.env is the single source of truth for runtime values — every model
identifier, base URL, fallback list, port, and credential MUST be
declared in `.env` (or `.env.example`). The defaults here are
intentionally empty so a missing/typo'd entry surfaces immediately
at startup via [Settings.validate_required] instead of silently
shipping a stale hardcoded value (e.g. last quarter's vision model).
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    """Environment variables for mint-agentic.

    All LLM-related fields default to "" / [] — populate them via
    `.env`. Required ones are checked at startup (see
    [Settings.validate_required]); optional ones (fallback lists,
    secondary classifier) stay empty when unset.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Provider credentials ─────────────────────────────────────────
    openrouter_api_key: str = ""
    typhoon_api_key: str = Field(default="", validation_alias="TYPOHOON_API_KEY")

    # ── Databases ────────────────────────────────────────────────────
    database_url: str = ""
    backend_database_url: str = ""

    # ── Studio + server ──────────────────────────────────────────────
    langgraph_studio_port: int = 8080
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # ── 1. ReAct (analytics chat) ────────────────────────────────────
    react_model: str = ""
    react_base_url: str = ""
    # Fallback active only when react_base_url is OpenRouter.
    react_fallback_models: list[str] = Field(default=[])
    # OpenRouter provider routing — opt-in per node. Empty list /
    # None = don't emit the `provider` block (default routing).
    react_provider_order: list[str] = Field(default=[])
    react_provider_allow_fallbacks: bool | None = None

    # ── 2. CodeAct (SQL/analytics sandbox) ───────────────────────────
    codeact_model: str = ""
    codeact_base_url: str = ""
    codeact_fallback_models: list[str] = Field(default=[])
    codeact_provider_order: list[str] = Field(default=[])
    codeact_provider_allow_fallbacks: bool | None = None

    # ── 3. Vision (slip / receipt parsing) ───────────────────────────
    # Must be a multimodal model that accepts `image_url` content blocks
    # in OpenAI-compatible chat format.
    vision_model: str = ""
    vision_base_url: str = ""
    # Pick fallbacks from a DIFFERENT vendor than the primary so a
    # provider-wide outage doesn't take the whole queue down. See the
    # comment in .env for known-bad models on slip OCR.
    vision_fallback_models: list[str] = Field(default=[])
    vision_provider_order: list[str] = Field(default=[])
    vision_provider_allow_fallbacks: bool | None = None

    # ── 4. Intent classifier (entry-router routing decision) ─────────
    intent_classifier_model: str = ""
    intent_classifier_base_url: str = ""
    # OpenRouter-side fallback queue. Same mechanism as React: each
    # entry is another model id, OpenRouter retries them in order
    # within a single HTTP call. Empty list = no fallback (primary
    # failure defaults to `intent='other'`, ReAct lane still works).
    intent_classifier_fallback_models: list[str] = Field(default=[])
    intent_classifier_provider_order: list[str] = Field(default=[])
    intent_classifier_provider_allow_fallbacks: bool | None = None

    # ── 5. Transaction (quick_add) ───────────────────────────────────
    # Must support tool/function calling AND Thai text output. Used by
    # `quick_add_node` to emit the `propose_transaction` tool call from
    # a free-form text journal entry. (The proposal save / dismiss ack
    # used to live here too but now ships from mobile via a local
    # template — see `POST /chat/intent`.)
    transaction_llm_model: str = ""
    transaction_llm_base_url: str = ""
    # Fallbacks MUST support tool calling (the `propose_transaction`
    # tool is the whole point of this node) AND produce Thai text for
    # the confirmation reply.
    transaction_llm_fallback_models: list[str] = Field(default=[])
    transaction_llm_provider_order: list[str] = Field(default=[])
    transaction_llm_provider_allow_fallbacks: bool | None = None

    # ── 6. Speech-to-text (/chat/voice) ──────────────────────────────
    # Multimodal model that accepts audio_input content blocks.
    stt_model: str = ""
    stt_base_url: str = ""
    # Fallback models MUST accept the `input_audio` content block —
    # most "multimodal" models on OpenRouter only handle images, not
    # audio. Verify support before adding.
    stt_fallback_models: list[str] = Field(default=[])
    stt_provider_order: list[str] = Field(default=[])
    stt_provider_allow_fallbacks: bool | None = None

    # ── Environment + logging ────────────────────────────────────────
    # Postel's-law fallbacks (sync/voice payload tolerance) only kick
    # in when production; dev/staging/test throw loudly so bad data
    # is caught before shipping.
    environment: Literal["development", "staging", "test", "production"] = "development"

    # `debug` = full node-internal traces (state, catalog counts,
    # compression). `production` = milestones only (HTTP entry, stream
    # lifecycle, tool boundaries with duration_ms, tool_calls summary).
    log_mode: Literal["production", "debug"] = "debug"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_debug_log(self) -> bool:
        return self.log_mode == "debug"

    # ── Validation ───────────────────────────────────────────────────
    # Required-field check runs at startup. Optional fields
    # (`*_fallback_models` lists, provider routing prefs) are
    # allowed to stay empty.
    _REQUIRED_FIELDS = (
        ("openrouter_api_key",        "OPENROUTER_API_KEY"),
        ("database_url",              "DATABASE_URL"),
        ("react_model",               "REACT_MODEL"),
        ("react_base_url",            "REACT_BASE_URL"),
        ("codeact_model",             "CODEACT_MODEL"),
        ("codeact_base_url",          "CODEACT_BASE_URL"),
        ("vision_model",              "VISION_MODEL"),
        ("vision_base_url",           "VISION_BASE_URL"),
        ("intent_classifier_model",   "INTENT_CLASSIFIER_MODEL"),
        ("intent_classifier_base_url","INTENT_CLASSIFIER_BASE_URL"),
        ("transaction_llm_model",     "TRANSACTION_LLM_MODEL"),
        ("transaction_llm_base_url",  "TRANSACTION_LLM_BASE_URL"),
        ("stt_model",                 "STT_MODEL"),
        ("stt_base_url",              "STT_BASE_URL"),
    )

    @model_validator(mode="after")
    def _validate_required(self) -> "Settings":
        """Fail loudly at import time when required `.env` entries are
        missing — a silent empty model would produce confusing 400s
        from the LLM provider at the first chat turn.

        The `*_FALLBACK_MODELS` lists are intentionally NOT required;
        an empty list simply disables OpenRouter's fallback queue for
        that node.
        """
        missing = [
            env for (attr, env) in self._REQUIRED_FIELDS
            if not (getattr(self, attr) or "").strip()
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Missing required environment variables in .env: {joined}.\n"
                f"Copy `.env.example` to `.env` and fill in the values."
            )
        return self


settings = Settings()
