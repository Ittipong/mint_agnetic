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


def create_stt_llm():
    """Speech-to-text LLM. Routes through OpenRouter so we get one bill
    and one auth flow for every multimodal call (vision, audio, text).

    Temperature pinned to 0 — transcription is a deterministic task; any
    randomness only introduces variance in the produced transcript.

    Gemini 2.0 Flash Lite accepts audio inputs via OpenRouter's
    OpenAI-compatible `input_audio` content block (see `stt_node.py`).
    """
    return _make_openrouter_llm(settings.stt_model, [], temperature=0.0)


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


def create_intent_classifier_fallback_llm():
    """Optional fallback LLM for the intent classifier.

    Returns None when no fallback model is configured — caller treats
    that as "no fallback available" and short-circuits to the existing
    `intent='other'` default. We keep this opt-in to avoid silently
    doubling per-classification cost (most workloads won't need it).

    When configured, the fallback should be a DIFFERENT provider than
    the primary so they don't share a single point of failure (one
    OpenRouter region down, one model deprecation, etc.).
    """
    model = (settings.intent_classifier_fallback_model or "").strip()
    if not model:
        return None
    if _is_openrouter(settings.intent_classifier_fallback_base_url):
        return _make_openrouter_llm(model, [], temperature=0.0)
    return _make_typhoon_llm(
        model,
        settings.intent_classifier_fallback_base_url,
        temperature=0.0,
    )


# ── Prompt caching helper ─────────────────────────────────────────────────────

# Models that benefit from explicit Anthropic-style cache_control via
# OpenRouter. DeepSeek and Gemini auto-cache by prefix without markers, so
# we don't wrap their messages (extra structured content blocks can confuse
# providers that don't honor cache_control).
_CACHE_CONTROL_MODELS = (
    "anthropic/",
    "amazon/nova",  # Nova models support cache_control via Bedrock
    "claude-",
)


def supports_cache_control(model: str | None) -> bool:
    """True when wrapping a SystemMessage with cache_control reduces cost."""
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) or p in m for p in _CACHE_CONTROL_MODELS)


def cached_system_content(text: str, model: str | None) -> str | list[dict]:
    """Return the right SystemMessage content shape for the target model.

    Plain string for DeepSeek/Gemini (auto prefix-cached by OpenRouter), or
    a structured content list with `cache_control={"type": "ephemeral"}` for
    models that require explicit caching markers (Anthropic, Nova). The list
    form is what `langchain_openai` forwards verbatim to OpenRouter, which
    passes the marker through to the upstream provider.
    """
    if not supports_cache_control(model):
        return text
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def create_transaction_llm():
    """LLM for the propose_transaction lanes (`quick_add` + `confirmation`).

    Picked separately from the main ReAct LLM so we can use a cheaper /
    faster model for the short, structured workload these nodes run:
    short prompt, one tool call (or one short text reply), no ReAct loop.

    Temperature 0.1 — we want deterministic wallet/category matches but
    leave a tiny bit of slack for the confirmation node's natural-language
    acknowledgement (e.g. "บันทึก KFC 100 บาทเรียบร้อย" vs "เก็บไว้แล้ว").
    """
    if _is_openrouter(settings.transaction_llm_base_url):
        return _make_openrouter_llm(
            settings.transaction_llm_model, [], temperature=0.1
        )
    return _make_typhoon_llm(
        settings.transaction_llm_model,
        settings.transaction_llm_base_url,
        temperature=0.1,
    )


# Module-level singletons — imported by nodes.py / step.py
llm = create_react_llm()
codeact_llm = create_codeact_llm()
vision_llm = create_vision_llm()
intent_classifier_llm = create_intent_classifier_llm()
# May be None — caller (classify_intent_node) checks before retry.
intent_classifier_fallback_llm = create_intent_classifier_fallback_llm()
stt_llm = create_stt_llm()
transaction_llm = create_transaction_llm()
