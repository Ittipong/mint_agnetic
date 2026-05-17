"""LLM setup for ReAct and CodeAct agents.

OpenRouter models go through `ChatOpenRouterREST` — our own
`BaseChatModel` subclass that POSTs directly to OpenRouter's REST
endpoint (see `llm_openrouter.py`). The direct path gives us full
control over the request body (`usage.include`, `models` fallback list,
`cache_control` content blocks) without fighting an SDK that doesn't
recognise OpenRouter-specific fields. The langchain-openrouter wrapper
was dropped after PoC (2026-05-17) confirmed its underlying openrouter
SDK rejects every escape hatch needed for cost surfacing.

Typhoon models still use `ChatOpenAI` — Typhoon is plain OpenAI-compat
with no cost field or fallback list, so the off-the-shelf wrapper fits.
"""

from langchain_openai import ChatOpenAI

from src.billing.callbacks import BillingCallback
from src.config import settings
from src.llm_openrouter import ChatOpenRouterREST


# Hard per-call timeout. Without this the SDKs wait forever on a stalled
# upstream connection and the whole SSE stream hangs (see logs around
# 2026-05-15T13:02 — second reason-turn LLM call never returned).
LLM_TIMEOUT_SEC = 60


def _is_openrouter(base_url: str) -> bool:
    return "openrouter" in base_url


def _make_openrouter_llm(
    model: str,
    fallback_models: list[str],
    temperature: float,
    *,
    base_url: str,
    feature: str = "chat",
    provider_order: list[str] | None = None,
    provider_allow_fallbacks: bool | None = None,
) -> ChatOpenRouterREST:
    return ChatOpenRouterREST(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=base_url,
        fallback_models=list(fallback_models),
        provider_order=list(provider_order or []),
        provider_allow_fallbacks=provider_allow_fallbacks,
        temperature=temperature,
        timeout=LLM_TIMEOUT_SEC,
        max_retries=2,
        # BillingCallback reads the per-request user_id from a
        # ContextVar set by server.py, extracts the cost from the
        # response, and fires a fire-and-forget POST to the Go backend.
        # See src/billing/__init__.py for the full pipeline.
        callbacks=[BillingCallback(feature=feature, model_hint=model)],
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
            settings.react_model,
            settings.react_fallback_models,
            temperature=0.3,
            base_url=settings.react_base_url,
            feature="chat",
            provider_order=settings.react_provider_order,
            provider_allow_fallbacks=settings.react_provider_allow_fallbacks,
        )
    return _make_typhoon_llm(settings.react_model, settings.react_base_url, temperature=0.3)


def create_codeact_llm():
    """CodeAct sandbox LLM. Fallback active only when using OpenRouter."""
    if _is_openrouter(settings.codeact_base_url):
        return _make_openrouter_llm(
            settings.codeact_model,
            settings.codeact_fallback_models,
            temperature=0.3,
            base_url=settings.codeact_base_url,
            feature="chat",
            provider_order=settings.codeact_provider_order,
            provider_allow_fallbacks=settings.codeact_provider_allow_fallbacks,
        )
    return _make_typhoon_llm(settings.codeact_model, settings.codeact_base_url, temperature=0.3)


def create_vision_llm():
    """Multimodal LLM for slip-to-transaction subgraph.

    Fallback active only when on OpenRouter — see .env for which
    models are known to fail on slip OCR (do NOT add them).
    """
    if _is_openrouter(settings.vision_base_url):
        return _make_openrouter_llm(
            settings.vision_model,
            settings.vision_fallback_models,
            temperature=0.1,
            base_url=settings.vision_base_url,
            feature="vision",
            provider_order=settings.vision_provider_order,
            provider_allow_fallbacks=settings.vision_provider_allow_fallbacks,
        )
    return _make_typhoon_llm(settings.vision_model, settings.vision_base_url, temperature=0.1)


def create_stt_llm():
    """Speech-to-text LLM. Routes through OpenRouter so we get one bill
    and one auth flow for every multimodal call (vision, audio, text).

    Temperature pinned to 0 — transcription is a deterministic task; any
    randomness only introduces variance in the produced transcript.

    Gemini 2.0 Flash Lite accepts audio inputs via OpenRouter's
    OpenAI-compatible `input_audio` content block (see `stt_node.py`).
    """
    return _make_openrouter_llm(
        settings.stt_model,
        settings.stt_fallback_models,
        temperature=0.0,
        base_url=settings.stt_base_url,
        feature="stt",
        provider_order=settings.stt_provider_order,
        provider_allow_fallbacks=settings.stt_provider_allow_fallbacks,
    )


def create_intent_classifier_llm():
    """Cheap, low-latency LLM for the entry-router intent classifier.

    Temperature pinned to 0 — classification is a discrete decision, not
    a creative task; randomness only adds variance to the routing.

    Fallback active only when on OpenRouter — Typhoon's OpenAI-compat
    endpoint doesn't support the `models` array.
    """
    if _is_openrouter(settings.intent_classifier_base_url):
        return _make_openrouter_llm(
            settings.intent_classifier_model,
            settings.intent_classifier_fallback_models,
            temperature=0.0,
            base_url=settings.intent_classifier_base_url,
            feature="chat",
            provider_order=settings.intent_classifier_provider_order,
            provider_allow_fallbacks=settings.intent_classifier_provider_allow_fallbacks,
        )
    return _make_typhoon_llm(
        settings.intent_classifier_model,
        settings.intent_classifier_base_url,
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
            settings.transaction_llm_model,
            settings.transaction_llm_fallback_models,
            temperature=0.1,
            base_url=settings.transaction_llm_base_url,
            feature="chat",
            provider_order=settings.transaction_llm_provider_order,
            provider_allow_fallbacks=settings.transaction_llm_provider_allow_fallbacks,
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
stt_llm = create_stt_llm()
transaction_llm = create_transaction_llm()
