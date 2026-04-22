"""OpenRouter LLM setup — uses LangChain OpenRouter integration.

Two models:
- writer_llm: Gemma for LLM Writer (friendly Thai responses)
- intent_llm: ministral-3b-2512 for Intent Router (classification)
"""

from langchain_openrouter import ChatOpenRouter

from src.config import settings


def create_writer_llm() -> ChatOpenRouter:
    """Create OpenRouter LLM with Gemma model for writer."""
    return ChatOpenRouter(
        model=settings.model,
        openrouter_api_key=settings.openrouter_api_key,
        base_url=settings.base_url,
    )


def create_intent_llm() -> ChatOpenRouter:
    """Create OpenRouter LLM with ministral for intent classification."""
    return ChatOpenRouter(
        model=settings.intent_model,
        openrouter_api_key=settings.openrouter_api_key,
        base_url=settings.base_url,
    )


# Default LLM (writer/gemma) — for backward compatibility
llm = create_writer_llm()

# Intent router LLM
intent_llm = create_intent_llm()
