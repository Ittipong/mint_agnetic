"""Minimal pytest fixtures + path setup for v3 Wave 1.

Heavier fixtures (mock LLM, mock proposal repo, mock tools) will be added
in later waves as more components come online. Wave 1 only exercises
sandbox-local utilities (safe_json, exceptions, schemas, state, validator)
and a single env-var contract for the DB pool — no shared mocks needed yet.

The `pythonpath = ["src", "tests_integration"]` block in pyproject.toml
already makes `src.agent…` importable; this file only adds defensive
fallback for direct invocations.
"""

from __future__ import annotations

import os
import sys

# Make `src` importable when pytest is run from the v3 root directly
# (matches the v2 conftest pattern). Also add the project root so
# `from src.agent...` imports resolve — `pythonpath` in pyproject only
# adds `src/`, which gives `agent.X`; tests in this tree use the
# explicit `src.agent.X` style to match production source imports.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Disable streaming preamble in tests — production UX feature that hits
# OpenRouter live; unit tests assert exact answer tokens and shouldn't
# depend on a real LLM round-trip. Real preamble behavior is covered by
# the live curl smoke test, not unit tests.
os.environ.setdefault("PREAMBLE_ENABLED", "0")

# server.py calls load_dotenv() at import, so the dev .env (router ON) leaked
# into every test collected after it and turned the classify router into a
# live LLM call. Pin it OFF here — load_dotenv never overrides an existing
# var; router tests opt in with monkeypatch.setenv.
os.environ["CLASSIFY_ROUTER_ENABLED"] = "0"


import pytest


@pytest.fixture(autouse=True)
def _hermetic_pair_resolver(monkeypatch):
    """Default stub for propose_transaction's joint wallet+category LLM call.

    Unstubbed it went to OpenRouter, so ADD tests passed or failed depending on
    whether a key leaked in from .env and on test order. Picks the first
    candidate wallet and a category whose name matches the hint (else None →
    Other-floor). Tests that need other behaviour monkeypatch over this.
    """
    import importlib

    pt = importlib.import_module("src.agent.tools.propose_transaction")

    async def fake_pair(*, candidate_wallets, candidate_cats_by_wallet,
                        category_hint=None, description=None, **_kw):
        if not candidate_wallets:
            return None
        w = candidate_wallets[0]
        hint = (category_hint or description or "").strip().lower()
        cat = next((c for c in candidate_cats_by_wallet.get(w.sync_id, [])
                    if hint and c.name.lower() == hint), None)
        return pt._PairChoice(wallet_sync_id=w.sync_id,
                              category_sync_id=cat.sync_id if cat else None,
                              confidence=1.0, reason="conftest stub")

    monkeypatch.setattr(pt, "_resolve_pair_async", fake_pair)
    yield
