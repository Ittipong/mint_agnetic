"""Test intent classification for Phase 1 intents."""

import pytest
from tests.evaluation.evaluators import Intent, TOOL_TO_INTENT, extract_intent_from_tools


class TestIntentClassification:
    """Tests for intent classification logic."""

    def test_tool_to_intent_mapping_complete(self):
        """Verify all 5 intents have tool mappings."""
        intents = {v for v in TOOL_TO_INTENT.values()}
        expected = {
            Intent.SPENT_BUDGET.value,
            Intent.WEEKLY_SUMMARY.value,
            Intent.DEBT_BALANCE.value,
            Intent.GOAL_PROGRESS.value,
            Intent.PAYDAY.value,
            Intent.FALLBACK.value,
        }
        assert intents == expected

    def test_extract_intent_from_tools_spent_budget(self):
        """get_spent_budget maps to SPENT_BUDGET."""
        tools = ["get_spent_budget"]
        assert extract_intent_from_tools(tools) == Intent.SPENT_BUDGET.value

    def test_extract_intent_from_tools_weekly_summary(self):
        """get_weekly_summary maps to WEEKLY_SUMMARY."""
        tools = ["get_weekly_summary"]
        assert extract_intent_from_tools(tools) == Intent.WEEKLY_SUMMARY.value

    def test_extract_intent_from_tools_debt_balance(self):
        """get_debt_balance maps to DEBT_BALANCE."""
        tools = ["get_debt_balance"]
        assert extract_intent_from_tools(tools) == Intent.DEBT_BALANCE.value

    def test_extract_intent_from_tools_goal_progress(self):
        """get_goal_progress maps to GOAL_PROGRESS."""
        tools = ["get_goal_progress"]
        assert extract_intent_from_tools(tools) == Intent.GOAL_PROGRESS.value

    def test_extract_intent_from_tools_payday(self):
        """get_payday maps to PAYDAY."""
        tools = ["get_payday"]
        assert extract_intent_from_tools(tools) == Intent.PAYDAY.value

    def test_extract_intent_from_tools_fallback(self):
        """get_recent_transactions maps to FALLBACK."""
        tools = ["get_recent_transactions"]
        assert extract_intent_from_tools(tools) == Intent.FALLBACK.value

    def test_extract_intent_from_tools_multiple(self):
        """When multiple tools, returns first match."""
        tools = ["get_spent_budget", "get_goal_progress"]
        assert extract_intent_from_tools(tools) == Intent.SPENT_BUDGET.value

    def test_extract_intent_from_tools_empty(self):
        """Empty tool list returns None."""
        assert extract_intent_from_tools([]) is None

    def test_extract_intent_from_tools_unknown(self):
        """Unknown tool returns None."""
        tools = ["some_unknown_tool"]
        assert extract_intent_from_tools(tools) is None


class TestIntentEnum:
    """Tests for Intent enum."""

    def test_all_intents_defined(self):
        """All 6 intents exist."""
        assert len(Intent) == 6

    def test_intent_values(self):
        """Verify intent string values."""
        assert Intent.SPENT_BUDGET.value == "SPENT_BUDGET"
        assert Intent.WEEKLY_SUMMARY.value == "WEEKLY_SUMMARY"
        assert Intent.DEBT_BALANCE.value == "DEBT_BALANCE"
        assert Intent.GOAL_PROGRESS.value == "GOAL_PROGRESS"
        assert Intent.PAYDAY.value == "PAYDAY"
        assert Intent.FALLBACK.value == "FALLBACK"
