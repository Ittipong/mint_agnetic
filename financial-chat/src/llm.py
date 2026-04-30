"""OpenRouter LLM setup — uses LangChain OpenRouter integration."""

import os

from langchain_openrouter import ChatOpenRouter

from src.config import settings

# LangChain's ChatOpenRouter reads OPENROUTER_API_KEY from os.environ directly.
# Set it here so the validation passes.
os.environ["OPENROUTER_API_KEY"] = settings.typhoon_api_key


def create_writer_llm() -> ChatOpenRouter:
    """Create OpenRouter LLM for ReAct reasoning (Typhoon)."""
    return ChatOpenRouter(
        model=settings.react_model,
        openrouter_api_key=settings.typhoon_api_key,
        base_url=settings.react_base_url,
    )


# Default LLM — used by reason_node in the ReAct loop
llm = create_writer_llm()
