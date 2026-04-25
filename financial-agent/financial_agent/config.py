from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/mint_money_dev"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash-lite"
    complex_model: str = "google/gemini-2.5-flash-lite"
    max_steps: int = 3
    executor_timeout: int = 30
    thinking_budget_tokens: int = 8000
    solve_timeout: int = 180
    log_level: str = "INFO"
    debug: bool = False


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
