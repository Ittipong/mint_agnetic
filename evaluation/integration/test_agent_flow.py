"""Integration tests for full agent flow."""

import pytest


class TestAgentFlow:
    """Tests for complete agent conversation flow."""

    @pytest.mark.asyncio
    async def test_agent_responds_to_spent_budget(self, studio_url, default_user_id):
        """Agent should respond to spent budget question."""
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{studio_url}/runs/wait",
                json={
                    "assistant_id": "agent",
                    "graph_name": "agent",
                    "input": {
                        "user_id": default_user_id,
                        "messages": [
                            {"role": "user", "content": "เดือนนี้ใช้ไปเท่าไร"}
                        ],
                    },
                },
            )
            assert response.status_code == 200
            data = response.json()

            # Should have messages
            assert "messages" in data
            messages = data["messages"]

            # Should have AI response
            ai_messages = [m for m in messages if m.get("type") == "ai"]
            assert len(ai_messages) > 0

            # Last AI message should have content
            last_ai = ai_messages[-1]
            assert last_ai.get("content")

    @pytest.mark.asyncio
    async def test_agent_handles_fallback(self, studio_url, default_user_id):
        """Agent should gracefully handle out-of-scope questions."""
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{studio_url}/runs/wait",
                json={
                    "assistant_id": "agent",
                    "graph_name": "agent",
                    "input": {
                        "user_id": default_user_id,
                        "messages": [
                            {"role": "user", "content": "ช่วยหาร้านอาหารอร่อยๆ หน่อย"}
                        ],
                    },
                },
            )
            assert response.status_code == 200
            data = response.json()

            # Should still respond
            messages = data["messages"]
            ai_messages = [m for m in messages if m.get("type") == "ai"]
            assert len(ai_messages) > 0

            # Should not call financial tools
            tool_calls = []
            for msg in messages:
                if tc := msg.get("tool_calls"):
                    tool_calls.extend(tc)

            # Fallback should not call financial tools
            tool_names = [tc.get("name") for tc in tool_calls]
            financial_tools = ["get_spent_budget", "get_weekly_summary", "get_debt_balance"]
            assert not any(t in tool_names for t in financial_tools)
