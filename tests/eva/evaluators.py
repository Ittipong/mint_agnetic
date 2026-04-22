"""Evaluators for AI Friend Phase 1.

Three evaluators:
  1. tool_selection   — code: did agent call the correct tool?
  2. no_shame         — code: no judgmental/shame language in response?
  3. response_quality — LLM-as-judge: Thai, helpful, uses real data, no hallucination?

Each function signature: (inputs, outputs, reference_outputs) -> dict
  inputs:            {"messages": [...], "user_id": "..."}
  outputs:           {"response": "...", "tool_calls": [...]}
  reference_outputs: {"intent": "...", "edge_case": "...", "description": "..."}
"""

from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter

from src.config import settings

# Maps expected intent → tools that confirm the intent was handled correctly
_INTENT_TOOLS: dict[str, list[str]] = {
    "SPENT_BUDGET": ["get_spent_budget"],
    "WEEKLY_SUMMARY": ["get_weekly_summary"],
    "DEBT_BALANCE": ["get_debt_balance"],
    "GOAL_PROGRESS": ["get_goal_progress"],
    "PAYDAY": ["get_payday"],
    "FALLBACK": [],
}

_TEMPLATE_TOOLS = {
    "get_spent_budget", "get_weekly_summary",
    "get_debt_balance", "get_goal_progress", "get_payday",
}

_SHAME_KEYWORDS = [
    "ใช้เยอะเกินไป",
    "ควรลด",
    "ไม่ควรซื้อ",
    "ใช้จ่ายมากเกินไป",
    "ฟุ่มเฟือย",
    "สิ้นเปลือง",
]

_QUALITY_PROMPT = """\
You are evaluating a Thai financial AI assistant named "Fin" in the Mint Money app.

Question: {question}
Expected intent: {intent}
Tools called: {tools}
Agent response:
{response}

ให้คะแนน 1 (ผ่าน) ถ้า ครบทุกข้อต่อไปนี้:
- ตอบเป็นภาษาไทย
- ตอบตรงกับคำถามที่ถาม
- ถ้ามีการเรียก tool ต้องใช้ข้อมูลจาก tool จริง ไม่แต่งตัวเลขเอง
- ใช้น้ำเสียงเป็นมิตร ไม่ตัดสินหรือตำหนิผู้ใช้

ให้คะแนน 0 (ไม่ผ่าน) ถ้า มีข้อใดข้อหนึ่งต่อไปนี้:
- ไม่ได้ตอบเป็นภาษาไทย
- ตอบนอกเรื่องหรือไม่ตรงคำถาม
- มีตัวเลขที่แต่งขึ้นเองโดยไม่มาจาก tool
- ไม่มีคำตอบหรือเกิด error

ตอบด้วย PASS หรือ FAIL บรรทัดแรก ตามด้วยอธิบายสั้นๆ เป็นภาษาไทย 1 ประโยค
"""


def evaluate_tool_selection(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:  # noqa: ARG001
    """Code-based: checks agent called the correct tool for the intent."""
    expected = reference_outputs.get("intent", "")
    called = set(outputs.get("tool_calls") or [])

    if expected == "FALLBACK":
        # FALLBACK must not trigger any template tool
        score = 0 if called & _TEMPLATE_TOOLS else 1
    else:
        expected_tools = set(_INTENT_TOOLS.get(expected, []))
        score = 1 if called & expected_tools else 0

    return {"key": "tool_selection", "score": score}


def evaluate_no_shame(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:  # noqa: ARG001
    """Code-based: response must not contain shame/judgment language."""
    response = outputs.get("response") or ""
    has_shame = any(kw in response for kw in _SHAME_KEYWORDS)
    return {"key": "no_shame", "score": 0 if has_shame else 1}


def evaluate_response_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: Thai, helpful, uses real data, no hallucination."""
    messages = inputs.get("messages") or [{}]
    question = messages[0].get("content", "") if messages else ""
    response = (outputs.get("response") or "")[:600]
    tool_calls = outputs.get("tool_calls") or []
    intent = reference_outputs.get("intent", "")

    prompt = _QUALITY_PROMPT.format(
        question=question,
        intent=intent,
        tools=", ".join(tool_calls) if tool_calls else "none",
        response=response,
    )

    judge = ChatOpenRouter(
        model=settings.intent_model,
        openrouter_api_key=settings.openrouter_api_key,
        base_url=settings.base_url,
    )
    result = judge.invoke([HumanMessage(content=prompt)])
    text = (result.content or "").strip()
    score = 1 if text.upper().startswith("PASS") else 0
    return {"key": "response_quality", "score": score, "comment": text[:200]}
