"""UT-CAP01..03 — `get_app_capability` tool sanity.

Replaces the inline APP CAPABILITIES block in the system prompt. Each
topic must carry the same shape so the LLM can compose redirects
predictably.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import ToolMessage

from src.agent.tools.app_capability import CAPABILITIES, get_app_capability


REQUIRED_KEYS = {
    "topic_label",
    "chat_can_do",
    "chat_cannot_do",
    "where_to_do_it",
    "phrase_examples",
    "redirect_text",
}

EXPECTED_TOPICS = {
    "edit_confirmed_txn", "delete_confirmed_txn", "backdate",
    "manage_wallet", "manage_budget", "manage_goal",
    "split_bill", "export_data", "recurring", "general",
}


def test_UT_CAP01_all_ten_topics_present():
    assert set(CAPABILITIES.keys()) == EXPECTED_TOPICS


def test_UT_CAP01_each_topic_structurally_complete():
    for topic, cap in CAPABILITIES.items():
        missing = REQUIRED_KEYS - set(cap.keys())
        assert not missing, f"{topic} missing {missing}"
        assert isinstance(cap["chat_can_do"], list)
        assert isinstance(cap["chat_cannot_do"], list)
        # Either we say what to do, or we explain the in-app destination.
        assert cap["chat_can_do"] or cap["chat_cannot_do"], (
            f"{topic}: both lists empty — no information for the LLM"
        )


def test_UT_CAP01_general_topic_lists_main_features():
    """The 'general' topic is the catch-all. It must mention ADD, backdate,
    analyst/advisor, and emotional support so an open question
    ('แอปนี้ทำอะไรได้บ้าง') gets a complete answer."""
    general = CAPABILITIES["general"]
    joined = " ".join(general["chat_can_do"]).lower()
    assert "บันทึก" in joined
    assert "ย้อนหลัง" in joined
    assert any(kw in joined for kw in ("วิเคราะห์", "สรุป", "ตอบ"))


@pytest.mark.asyncio
async def test_UT_CAP02_tool_invocation_returns_json_toolmessage():
    result = await get_app_capability.ainvoke({
        "name": "get_app_capability",
        "args": {"topic": "delete_confirmed_txn"},
        "type": "tool_call",
        "id": "tc-test",
    })
    assert isinstance(result, ToolMessage)
    payload = json.loads(result.content)
    assert payload["topic"] == "delete_confirmed_txn"
    assert payload["redirect_text"]  # this topic has a canned redirect


def test_UT_CAP03_literal_type_matches_capability_keys():
    schema = get_app_capability.args_schema.model_json_schema()
    topic_prop = schema["properties"]["topic"]
    enum_values = set(topic_prop.get("enum", []))
    assert enum_values == EXPECTED_TOPICS, (
        f"Literal[topic] drift — enum {enum_values} vs "
        f"CAPABILITIES {EXPECTED_TOPICS}"
    )
