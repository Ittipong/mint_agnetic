"""LLM setup for ReAct and CodeAct agents.

OpenRouter models use ChatOpenRouter (native SDK, cost tracking, clean fallback).
Typhoon models use ChatOpenAI (OpenAI-compatible endpoint, no fallback support).
"""

from langchain_openai import ChatOpenAI

from src.config import settings


# Hard per-call timeout. Without this the SDKs wait forever on a stalled
# upstream connection and the whole SSE stream hangs (see logs around
# 2026-05-15T13:02 — second reason-turn LLM call never returned).
LLM_TIMEOUT_SEC = 60


def _is_openrouter(base_url: str) -> bool:
    return "openrouter" in base_url


def _make_openrouter_llm(model: str, fallback_models: list[str], temperature: float):
    from langchain_openrouter import ChatOpenRouter  # noqa: PLC0415

    model_kwargs: dict = {}
    if fallback_models:
        model_kwargs["models"] = [model, *fallback_models]

    # ChatOpenRouter takes `timeout` in **milliseconds** (maps to SDK timeout_ms).
    return ChatOpenRouter(
        model=model,
        api_key=settings.openrouter_api_key,
        temperature=temperature,
        timeout=LLM_TIMEOUT_SEC * 1000,
        max_retries=2,
        model_kwargs=model_kwargs,
    )


def _make_typhoon_llm(model: str, base_url: str, temperature: float) -> ChatOpenAI:
    # ChatOpenAI takes `timeout` in **seconds** (float|int).
    return ChatOpenAI(
        model=model,
        api_key=settings.typhoon_api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=LLM_TIMEOUT_SEC,
        max_retries=2,
    )


def create_react_llm():
    """ReAct reasoning LLM. Fallback active only when using OpenRouter."""
    if _is_openrouter(settings.react_base_url):
        return _make_openrouter_llm(
            settings.react_model, settings.react_fallback_models, temperature=0.3
        )
    return _make_typhoon_llm(settings.react_model, settings.react_base_url, temperature=0.3)


def create_codeact_llm():
    """CodeAct sandbox LLM. Fallback active only when using OpenRouter."""
    if _is_openrouter(settings.codeact_base_url):
        return _make_openrouter_llm(
            settings.codeact_model, settings.codeact_fallback_models, temperature=0.3
        )
    return _make_typhoon_llm(settings.codeact_model, settings.codeact_base_url, temperature=0.3)


def create_vision_llm():
    """Multimodal LLM for slip-to-transaction subgraph."""
    if _is_openrouter(settings.vision_base_url):
        return _make_openrouter_llm(settings.vision_model, [], temperature=0.1)
    return _make_typhoon_llm(settings.vision_model, settings.vision_base_url, temperature=0.1)


def create_intent_classifier_llm():
    """Cheap, low-latency LLM for the entry-router intent classifier.

    Temperature pinned to 0 — classification is a discrete decision, not
    a creative task; randomness only adds variance to the routing.
    """
    if _is_openrouter(settings.intent_classifier_base_url):
        return _make_openrouter_llm(
            settings.intent_classifier_model, [], temperature=0.0
        )
    return _make_typhoon_llm(
        settings.intent_classifier_model,
        settings.intent_classifier_base_url,
        temperature=0.0,
    )


# Module-level singletons — imported by nodes.py / step.py
llm = create_react_llm()
codeact_llm = create_codeact_llm()
vision_llm = create_vision_llm()
intent_classifier_llm = create_intent_classifier_llm()
