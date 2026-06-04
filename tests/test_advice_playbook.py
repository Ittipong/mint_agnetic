"""UT-AP01..03 — `get_advice_playbook` tool sanity.

The advisor playbook is a static-content tool: the LLM calls it to load
decision frameworks for major-life finance topics (home, refi, car, debt,
tax, invest, discipline). These tests guard:
  AP01 — all 7 topics are present and structurally complete.
  AP02 — tool invocation returns a ToolMessage with parseable JSON.
  AP03 — Literal typing on `topic` is in lockstep with PLAYBOOKS keys.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import ToolMessage

from src.agent.tools.advice_playbook import (
    PLAYBOOKS,
    get_advice_playbook,
)


REQUIRED_KEYS = {
    "topic_label",
    "tone_note",
    "discovery_checklist",
    "framework",
    "red_flags",
}

EXPECTED_TOPICS = {
    "home", "refinance", "car", "debt", "tax", "invest", "discipline",
}


def test_UT_AP01_all_seven_topics_present():
    """Every topic from R11 must have a playbook."""
    assert set(PLAYBOOKS.keys()) == EXPECTED_TOPICS, (
        f"PLAYBOOKS drift — expected {EXPECTED_TOPICS}, "
        f"got {set(PLAYBOOKS.keys())}"
    )


def test_UT_AP01_each_topic_structurally_complete():
    """Every playbook must carry the required sections so the LLM has a
    consistent shape to reason against."""
    for topic, pb in PLAYBOOKS.items():
        missing = REQUIRED_KEYS - set(pb.keys())
        assert not missing, f"{topic} missing keys {missing}"
        assert isinstance(pb["discovery_checklist"], list)
        assert len(pb["discovery_checklist"]) >= 3, (
            f"{topic}: discovery_checklist too short — "
            f"R11/R12 needs enough items to ask incrementally"
        )
        assert isinstance(pb["framework"], dict)
        assert pb["framework"], f"{topic}: framework dict is empty"
        assert isinstance(pb["red_flags"], list)
        assert pb["red_flags"], f"{topic}: red_flags is empty"


def test_UT_AP01_invest_has_prerequisite_check():
    """Investing without an emergency fund / high-rate debt cleared is
    irresponsible advice. The invest playbook must remind the LLM to gate
    on this before recommending allocation."""
    invest = PLAYBOOKS["invest"]
    assert "prerequisite_check" in invest
    assert any(
        "emergency" in s.lower() for s in invest["prerequisite_check"]
    ), "invest prereq must mention emergency fund"
    assert any(
        "debt" in s.lower() or "หนี้" in s for s in invest["prerequisite_check"]
    ), "invest prereq must mention high-rate debt"


@pytest.mark.asyncio
async def test_UT_AP02_tool_invocation_returns_json_toolmessage():
    """Tool returns a ToolMessage whose `content` is JSON containing the
    full playbook plus a `topic` field echoed back."""
    result = await get_advice_playbook.ainvoke({
        "name": "get_advice_playbook",
        "args": {"topic": "debt"},
        "type": "tool_call",
        "id": "tc-test",
    })

    assert isinstance(result, ToolMessage)
    payload = json.loads(result.content)
    assert payload["topic"] == "debt"
    assert payload["topic_label"]
    assert "framework" in payload
    # Thai content must survive JSON round-trip
    assert "หนี้" in payload["topic_label"]


def test_UT_AP03_literal_type_matches_playbook_keys():
    """The Literal[...] annotation on `topic` constrains the LLM at the
    tool-binding layer. If a topic is added to PLAYBOOKS without
    updating Literal in the signature, the LLM cannot call it. Guard
    the lockstep by inspecting the tool's generated args schema."""
    schema = get_advice_playbook.args_schema.model_json_schema()
    # Pydantic v2 inlines Literal as enum on the property
    topic_prop = schema["properties"]["topic"]
    enum_values = set(topic_prop.get("enum", []))
    assert enum_values == EXPECTED_TOPICS, (
        f"Literal[topic] drift — enum has {enum_values}, "
        f"PLAYBOOKS has {EXPECTED_TOPICS}"
    )
