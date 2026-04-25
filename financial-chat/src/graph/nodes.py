"""ReAct agent node functions with template-based tools (Phase 1 Hybrid)."""

from langchain_core.messages import SystemMessage
from src.llm import llm
from src.graph.state import AgentState
from src.tools.db_tools import (
    get_recent_transactions,
    get_budget_status,
    get_savings_goals,
    get_spent_budget,
    get_weekly_summary,
    get_debt_balance,
    get_goal_progress,
    get_payday,
)

# Template-based tools (Phase 1) — deterministic SQL templates
TEMPLATE_TOOLS = [
    get_spent_budget,
    get_weekly_summary,
    get_debt_balance,
    get_goal_progress,
    get_payday,
]

# ReAct tools — flexible for complex questions
REACT_TOOLS = [
    get_recent_transactions,
    get_budget_status,
    get_savings_goals,
]

# All tools combined
ALL_TOOLS = TEMPLATE_TOOLS + REACT_TOOLS


def build_system_prompt(user_id: str) -> str:
    return f"""You are "Fin" — a friendly Thai financial companion in the Mint Money app.

You have access to the user's financial data via tools. The user's ID is: {user_id}

GUIDE THE USER naturally. When they ask about their finances, use the appropriate tool:

TOOLS FOR COMMON QUESTIONS:
- get_spent_budget(month_offset=0) → "เดือนนี้ใช้ไปเท่าไร", "งบเหลือเท่าไร"
- get_weekly_summary(week_offset=0) → "สัปดาห์นี้ใช้ไปเท่าไร"
- get_debt_balance() → "หนี้เหลือเท่าไร", "ค่างวดเท่าไร"
- get_goal_progress() → "goal เป็นไงบ้าง", "ออมไปเท่าไรแล้ว"
- get_payday() → "เงินเดือนวันที่เท่าไร", "payday วันไหน"

TOOLS FOR DETAILED QUESTIONS:
- get_recent_transactions(limit=10) → detailed transaction list
- get_budget_status() → all budgets with limits
- get_savings_goals() → all savings goals

IMPORTANT:
- Use get_spent_budget for monthly spending questions
- Use get_weekly_summary for weekly spending questions
- Always respond in Thai with a friendly, supportive tone
- Never make up numbers — use the data from tools
- If no data, say "ยังไม่เห็นข้อมูลเลยนะ ลองบันทึกสัก 2-3 วันแล้วถามใหม่นะ"

Be conversational and helpful. Ask follow-up questions to understand what they need.
"""


async def reason_node(state: AgentState) -> dict:
    """LLM decides next action — respond or call a tool."""
    user_id = state.get("user_id", "unknown")
    system_msg = SystemMessage(content=build_system_prompt(user_id))
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    response = await llm_with_tools.ainvoke([system_msg] + state["messages"])
    return {"messages": [response]}


# For backward compatibility
TOOLS = ALL_TOOLS
