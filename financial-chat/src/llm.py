"""LLM setup for ReAct agent."""

import os

from langchain_openai import ChatOpenAI

from src.config import settings


def create_writer_llm() -> ChatOpenAI:
    """Create Typhoon LLM for ReAct reasoning using OpenAI-compatible client."""
    return ChatOpenAI(
        model=settings.react_model,
        api_key=settings.typhoon_api_key,
        base_url=settings.react_base_url,
        temperature=0.7,
    )


# Default LLM — used by reason_node in the ReAct loop
llm = create_writer_llm()
