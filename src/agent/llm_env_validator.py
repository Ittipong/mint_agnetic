"""Startup validator for LLM environment variables.

Called once from the server lifespan BEFORE the first request is served.
Fails loud with a single error message that lists EVERY violation at once
(not first-fail), so a misconfigured deployment surfaces all issues in one
boot iteration instead of N restarts.

Two checks:

1. Deprecated guard — these vars used to exist but have been removed. If
   any are still set in the environment, refuse to start so a stale config
   can't leak to production silently.

2. Pairing rule — for every active role (see `llm_openrouter._MODEL_ENV`):
     - `<ROLE>_MODEL` MUST be set
     - `<ROLE>_FALLBACK_MODELS` MUST be set
     - `<ROLE>_FALLBACK_MODELS` MUST parse to ≥ 2 model slugs

   Rationale: every primary model has at least 2 fallbacks so a single
   upstream provider hiccup can't kill a turn. All slugs route through
   OpenRouter (single API key, single base URL).
"""

from __future__ import annotations

import os

from src.agent.llm_openrouter import (
    _FALLBACK_ENV,
    _MODEL_ENV,
    _resolve_fallbacks,
)


# Vars that previously existed but have been intentionally removed. If a
# deployment still sets one of these, the operator likely has stale config
# expectations and we MUST fail loud rather than silently ignore.
_DEPRECATED_ENV_VARS: tuple[str, ...] = (
    # Removed roles (no code path consumes these anymore).
    "TRANSACTION_LLM_MODEL",
    "TRANSACTION_LLM_BASE_URL",
    "TRANSACTION_LLM_FALLBACK_MODELS",
    "INTENT_CLASSIFIER_MODEL",
    "INTENT_CLASSIFIER_BASE_URL",
    "INTENT_CLASSIFIER_FALLBACK_MODELS",
    # Symmetric-fallback escape hatch — every role now requires its own
    # `<ROLE>_MODEL`, no cross-role implicit chains.
    "UNDERSTAND_MODEL",
    # Removed alternative provider — all routing is via OpenRouter.
    "TYPOHOON_API_KEY",
    # Voice merged STT+classify fast path — DROPPED in v3 (pure ReAct
    # does classification implicitly through tool calls). See
    # voice_handler.py:5-11 for the rationale.
    "VOICE_FAST_PATH",
    "VOICE_UNDERSTAND_MODEL",
    "VOICE_UNDERSTAND_BASE_URL",
    "VOICE_UNDERSTAND_FALLBACK_MODELS",
    # Within-thread episodic memory (extract/compress) — lived only in v2's
    # Understand/finalize nodes. v3 is pure ReAct with no understand.py and
    # no episodes module; no code path ever invoked these roles.
    "EPISODE_EXTRACT_MODEL",
    "EPISODE_EXTRACT_FALLBACK_MODELS",
    "EPISODE_COMPRESS_MODEL",
    "EPISODE_COMPRESS_FALLBACK_MODELS",
    # Cross-thread LangMem fact promoter — its only consumer
    # (long_term_memory.py) piggybacked on v2's compression event, which
    # doesn't exist in v3. The module is unwired and was deleted; this role
    # has no live caller. (LANGMEM_EMBED / LANGMEM_EMBED_DIMS were read by
    # store_factory, which was deleted when memory_recall/memory_write were
    # retired — they now do nothing either, but were never shipped in .env so
    # they are not worth a fail-loud guard.)
    "LANGMEM_MODEL",
    "LANGMEM_FALLBACK_MODELS",
)

_MIN_FALLBACKS = 2


class LLMEnvConfigError(RuntimeError):
    """Raised when the LLM environment fails validation at startup."""


def validate_llm_env() -> None:
    """Validate every known LLM role's env contract.

    Collects ALL violations before raising — operators see the full picture
    in one log line instead of fixing-restarting-fixing.

    Raises:
        LLMEnvConfigError: if any deprecated var is set, OR any role is
            missing its `<ROLE>_MODEL`, OR any `<ROLE>_FALLBACK_MODELS`
            has fewer than `_MIN_FALLBACKS` entries.
    """
    violations: list[str] = []

    # 1. Deprecated guard.
    for var in _DEPRECATED_ENV_VARS:
        if os.getenv(var):
            violations.append(
                f"[deprecated] env var {var!r} is set but has been removed. "
                f"Delete it from your .env / deployment config."
            )

    # 2. Pairing + ≥ _MIN_FALLBACKS rule. Iterate over the canonical role
    # list so a typo'd env var (e.g. PREAMBL_MODEL) gets flagged via the
    # "primary missing" check rather than silently passing.
    for role, model_env in _MODEL_ENV.items():
        if not os.getenv(model_env):
            violations.append(
                f"[role={role}] primary env {model_env!r} is not set"
            )
            continue  # no point checking fallback if primary is missing

        fb_env = _FALLBACK_ENV.get(role)
        if not fb_env:
            violations.append(
                f"[role={role}] no _FALLBACK_ENV mapping registered — "
                f"open llm_openrouter.py and add it"
            )
            continue

        fb_raw = os.getenv(fb_env, "").strip()
        if not fb_raw:
            violations.append(
                f"[role={role}] fallback env {fb_env!r} is not set "
                f"(required: JSON array of ≥{_MIN_FALLBACKS} OpenRouter slugs)"
            )
            continue

        fb_list = _resolve_fallbacks(role)
        if len(fb_list) < _MIN_FALLBACKS:
            violations.append(
                f"[role={role}] fallback env {fb_env!r} has {len(fb_list)} "
                f"model(s); minimum is {_MIN_FALLBACKS} so a single upstream "
                f"hiccup cannot kill the turn. Current value: {fb_raw!r}"
            )

    if violations:
        joined = "\n  - ".join(violations)
        raise LLMEnvConfigError(
            f"LLM env validation FAILED ({len(violations)} issue(s)):\n  - {joined}"
        )


__all__ = ["validate_llm_env", "LLMEnvConfigError"]
