"""ReAct agent node functions."""

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from src.llm import llm
from src.graph.state import AgentState
from src.tools.financial_info import get_financial_advice
from src.graph.codeact_subgraph import CODEACT_TOOL_NAME


# ── Tool declarations ────────────────────────────────────────────────────────

@tool
async def analyze_user_finances(task: str) -> str:
    """Fetch and analyze the user's real financial data using AI code execution.

    Use this for any question that requires real numbers from the database:
    transactions, budgets, balances, goals, debt, income, spending patterns.

    Args:
        task: Financial analysis task — MUST be written in English only.
              Be specific: include what data to fetch, date ranges, and aggregations.
              Example: "Get all savings goals with target amount, current balance,
                        target date, and monthly required savings"

    Returns:
        Analysis result with data, breakdown, confidence, and caveats.
    """
    # Execution is handled by codeact_node in the graph — this body is never reached.
    # The tool declaration exists solely so the LLM knows the schema.
    raise NotImplementedError("Routed to codeact_node by the graph")


# All tools ReAct knows about (bound to LLM for schema)
ALL_TOOLS = [
    analyze_user_finances,
    get_financial_advice,
]

# Tool names handled by codeact_node instead of ToolNode
CODEACT_TOOL_NAMES = {CODEACT_TOOL_NAME}

# Tools executed by the regular ToolNode
REGULAR_TOOLS = [t for t in ALL_TOOLS if t.name not in CODEACT_TOOL_NAMES]


# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(user_id: str) -> str:
    return f"""You are "Fin" — a friendly Thai financial companion in the Mint Money app.
User ID: {user_id}

TOOLS:
- analyze_user_finances(task: str) → fetch and analyze real financial data (transactions, budget, balance, goals, debt, income)
- get_financial_advice(topic: str) → curated strategies: debt_snowball, debt_avalanche, credit_card_trap, emergency_fund, 50_30_20_rule, saving_strategies, compound_interest, budget_basics

MULTI-TOOL STRATEGY:
When the user's question requires BOTH data AND advice, call them in SEPARATE steps (never in parallel):

Example — "ดู goal แล้วช่วยวางแผนว่าต้องออมเดือนละเท่าไร":
  Step 1: analyze_user_finances("Get all savings goals: name, target amount, current balance, target date, monthly required savings")
  Step 2: get_financial_advice("saving_strategies")
  Step 3: Combine → show goal data + tailored savings plan

Example — "มีหนี้บัตรเครดิต ควรใช้กลยุทธ์อะไร":
  Step 1: analyze_user_finances("Get all credit card wallets: name, outstanding balance, credit limit, due day")
  Step 2: get_financial_advice("credit_card_trap")
  Step 3: Combine → show actual debt numbers + strategy recommendation

RULES:
1. analyze_user_finances task MUST be in English only — never Thai
2. Never call analyze_user_finances and other tools in the same step — always call them separately
3. Never invent numbers — always fetch from analyze_user_finances first
4. Reply to user in Thai, friendly tone
5. If no data: "ยังไม่เห็นข้อมูลเลยนะ ลองบันทึกสัก 2-3 วันแล้วถามใหม่นะ"
6. Ask follow-up questions to understand the user's real situation
"""


# ── Reason node ───────────────────────────────────────────────────────────────

async def reason_node(state: AgentState) -> dict:
    """LLM decides next action — respond directly or call a tool."""
    user_id = state.get("user_id", "unknown")
    system_msg = SystemMessage(content=build_system_prompt(user_id))
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    response = await llm_with_tools.ainvoke([system_msg] + state["messages"])
    return {"messages": [response]}


# Legacy alias — kept so any existing import of TOOLS still works
TOOLS = ALL_TOOLS
