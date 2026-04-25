"""Basic smoke tests."""

from src.graph.state import AgentState
from src.graph.nodes import TOOLS


class TestToolsAvailable:
    """Verify tools are correctly defined."""

    def test_has_three_tools(self):
        assert len(TOOLS) == 3

    def test_tool_names(self):
        names = [t.name for t in TOOLS]
        assert "get_recent_transactions" in names
        assert "get_budget_status" in names
        assert "get_savings_goals" in names


class TestStateSchema:
    """Verify state schema is correct."""

    def test_agent_state_fields(self):
        state = AgentState(messages=[], user_id="test-user")
        assert "messages" in state
        assert "user_id" in state
