"""LLM setup for ReAct agent."""

import os

from langchain_openai import ChatOpenAI

from src.config import settings


def create_writer_llm() -> ChatOpenAI:
    """Create LLM for ReAct reasoning using OpenAI-compatible client."""
    # Use OpenRouter API key when using OpenRouter base URL
    if "openrouter" in settings.react_base_url:
        api_key = settings.openrouter_api_key
    else:
        api_key = settings.typhoon_api_key

    return ChatOpenAI(
        model=settings.react_model,
        api_key=api_key,
        base_url=settings.react_base_url,
        temperature=0.3,
    )


# Default LLM — used by reason_node in the ReAct loop
llm = create_writer_llm()
