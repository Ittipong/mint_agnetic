"""Langfuse tracing helper (Phase-1 PoC) — opt-in, graceful, non-mutating.

The handler must NEVER change the chat path when disabled, must never raise, and
must stamp the trace with the user + session so a turn is traceable per user.
"""

from __future__ import annotations

import importlib

import pytest

lf = importlib.import_module("src.agent.observability.langfuse_tracing")


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    # Each test resolves the handler fresh (module caches it after first build).
    monkeypatch.setattr(lf, "_resolved", False)
    monkeypatch.setattr(lf, "_handler", None)


def _cfg() -> dict:
    return {"configurable": {"thread_id": "t1"}, "recursion_limit": 25}


def test_disabled_is_exact_noop(monkeypatch):
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    cfg = _cfg()
    out = lf.attach_langfuse(cfg, user_id="u1", session_id="t1", tags=["chat"])
    assert out == cfg
    assert "callbacks" not in out and "metadata" not in out


def test_enabled_without_package_degrades(monkeypatch):
    # Force the handler build to fail (as if langfuse isn't installed) -> no-op.
    monkeypatch.setenv("LANGFUSE_ENABLED", "on")
    monkeypatch.setattr(lf, "_build_handler", lambda: None)
    cfg = _cfg()
    out = lf.attach_langfuse(cfg, user_id="u1", session_id="t1")
    assert out == cfg  # unchanged, no crash


def test_enabled_merges_handler_and_trace_attrs(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "on")
    sentinel = object()
    monkeypatch.setattr(lf, "_build_handler", lambda: sentinel)
    cfg = _cfg()
    out = lf.attach_langfuse(cfg, user_id="u1", session_id="t1", tags=["chat"])

    assert out["callbacks"] == [sentinel]
    assert out["metadata"] == {
        "langfuse_user_id": "u1",
        "langfuse_session_id": "t1",
        "langfuse_tags": ["chat"],
    }
    # The existing configurable/recursion_limit survive.
    assert out["configurable"] == {"thread_id": "t1"}
    assert out["recursion_limit"] == 25
    # Input config is not mutated (a shared default must stay clean).
    assert "callbacks" not in cfg and "metadata" not in cfg


def test_enabled_preserves_existing_callbacks_and_metadata(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "on")
    sentinel = object()
    existing_cb = object()
    monkeypatch.setattr(lf, "_build_handler", lambda: sentinel)
    cfg = {**_cfg(), "callbacks": [existing_cb], "metadata": {"keep": 1}}
    out = lf.attach_langfuse(cfg, user_id="u1", session_id="t1")

    assert out["callbacks"] == [existing_cb, sentinel]  # appended, not replaced
    assert out["metadata"]["keep"] == 1                  # existing metadata kept
    assert out["metadata"]["langfuse_user_id"] == "u1"


def test_never_raises(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "on")

    def _boom():
        raise RuntimeError("handler build blew up")

    monkeypatch.setattr(lf, "_build_handler", _boom)
    cfg = _cfg()
    # _get_handler raises inside attach_langfuse -> swallowed, config returned.
    out = lf.attach_langfuse(cfg, user_id="u1", session_id="t1")
    assert out == cfg
