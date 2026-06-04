"""`get_advice_playbook` — load decision framework for major-life finance topics.

NEW in Wave 5 (Advisor expansion).

The system prompt keeps emotional-support rules (R12, P0 EMOTIONAL FIRST-AID)
inline because they can fire on any turn. The detailed decision frameworks
for HOME / REFINANCE / CAR / DEBT / TAX / INVEST / DISCIPLINE are large
(~1.5k tokens combined) and only matter when the user is actually engaging
on that topic (<10% of turns). Splitting them into a tool keeps the default
prompt lean and lets the LLM load just-in-time context.

Tool surface:
  topic ∈ {"home", "refinance", "car", "debt", "tax", "invest", "discipline"}
  returns dict with: discovery_checklist, framework, red_flags, tone_note

The returned dict is opinion-laden by design — Mint Money's job is to give
a stance, not survey-style "consider these factors". User can always push
back; the LLM will adapt. R8 disclaimer still binds.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.session_logger import slog


_STATUS_WORD = "กำลังเตรียมข้อมูลให้..."


# ── Playbook content (static, no I/O) ──────────────────────────────────────

PLAYBOOKS: dict[str, dict] = {
    "home": {
        "topic_label": "ซื้อบ้าน / ผ่อนบ้าน",
        "tone_note": (
            "Buying a home is often the largest decision of someone's life. "
            "Be opinionated but kind. If numbers look tight, say so plainly — "
            "do NOT cheerlead someone into a 30-year mistake."
        ),
        "discovery_checklist": [
            "ราคาบ้านที่หมายตา + ทำเลคร่าวๆ",
            "Timeline — จะซื้อภายในกี่เดือน/ปี",
            "อายุงาน + อาชีพ (มนุษย์เงินเดือน / freelance / เจ้าของกิจการ)",
            "รายได้ net ต่อเดือน (หลังภาษี + ประกันสังคม)",
            "ภาระผ่อนอื่นต่อเดือน (บัตร / รถ / กยศ / หนี้ทุกชนิด)",
            "เงินเก็บที่เตรียมใช้ดาวน์ + ค่าโอน/ตกแต่ง",
        ],
        "framework": {
            "DTI_rule": "Total monthly debt service ≤ 40% of net income",
            "house_payment_ceiling": "House payment alone ≤ 30% of net income",
            "min_downpayment": (
                "First home: 10% (per BoT LTV); second home: 20%. "
                "Plus 5-7% buffer for transfer/legal/furnishing."
            ),
            "stress_test": (
                "Recompute payment at current rate + 2%. If still affordable, "
                "proceed. If not, lower the budget."
            ),
            "price_rule_of_thumb": (
                "Safe: house price ≤ 5× annual income. "
                "Tight: 5-7×. Dangerous: > 7×."
            ),
            "term_choice": (
                "Longer term lowers monthly payment but extends interest cost. "
                "Prefer the shortest term you can afford comfortably."
            ),
        },
        "red_flags": [
            "DTI > 50% — too tight, one shock breaks it",
            "Down payment empties savings to zero",
            "Job tenure < 1 year — banks may decline anyway",
            "Income is variable (commission/freelance) + long fixed payment",
            "User has no emergency fund AND is buying with full down",
        ],
    },

    "refinance": {
        "topic_label": "รีไฟแนนซ์บ้าน",
        "tone_note": (
            "Most users overestimate refi savings and underestimate fees. "
            "Walk them through the actual math before recommending."
        ),
        "discovery_checklist": [
            "ผ่อนบ้านมากี่ปีแล้ว + เหลืออีกกี่ปี",
            "ยอดหนี้คงเหลือปัจจุบัน",
            "ดอกเบี้ยปัจจุบัน (%) — อยู่ในช่วงโปรหรือหลังหมดโปร",
            "ค่างวดต่อเดือนปัจจุบัน",
            "แผนจะอยู่บ้านนี้อีกกี่ปี",
        ],
        "framework": {
            "worth_it_threshold": (
                "Rate drop ≥ 0.5-1% AND you'll stay in the home long enough "
                "to recoup refi fees"
            ),
            "payback_period": (
                "payback_months = total_refi_fees / monthly_savings. "
                "Must be < remaining residency."
            ),
            "refi_fees_estimate": (
                "1-3% of remaining principal — mortgage registration, "
                "appraisal, fire insurance, MRTA top-up"
            ),
            "retention_first": (
                "ALWAYS try the current bank's retention rate FIRST. "
                "It's cheaper (no registration fee) and faster. "
                "Only refi to another bank if retention won't match."
            ),
            "timing_rhythm": (
                "Most home loans have a 3-year low-rate teaser then revert "
                "to MRR/MLR. Plan to refi/retention at year 3, 6, 9..."
            ),
        },
        "red_flags": [
            "Planning to sell within 2 years — refi rarely pays back",
            "Existing loan has prepayment penalty still active",
            "User chasing 0.25% drop while ignoring 80k in fees",
        ],
    },

    "car": {
        "topic_label": "ซื้อรถ / ผ่อนรถ",
        "tone_note": (
            "Cars are depreciating assets. If the user is excited, ground "
            "them in the 20/4/10 rule before they sign anything."
        ),
        "discovery_checklist": [
            "ใช้รถทำอะไร (ทำงาน / ครอบครัว / ทำกิน)",
            "รถที่สนใจ + ราคา + ใหม่หรือมือสอง",
            "รายได้ net ต่อเดือน",
            "ภาระผ่อนอื่น",
            "เงินดาวน์ที่เตรียมไว้",
        ],
        "framework": {
            "20_4_10_rule": (
                "Down payment ≥ 20%, loan term ≤ 4 years, "
                "total car cost (payment + fuel + insurance + maintenance) "
                "≤ 10% of monthly net income"
            ),
            "new_vs_used": (
                "New car loses 15-20% value in year 1. "
                "Used (2-3 year old) is the sweet spot for depreciation/reliability."
            ),
            "true_cost": (
                "Don't compare ONLY car prices. Add: insurance "
                "(~3-5% of car price/yr), fuel, maintenance, parking, tax. "
                "Often 50% more than the loan payment."
            ),
        },
        "red_flags": [
            "0% down + 7-year loan — long-tail interest trap",
            "Car price > annual income — sinks net worth",
            "Buying new luxury car while carrying credit card debt",
            "Financing through dealer at high rate without comparing banks",
        ],
    },

    "debt": {
        "topic_label": "แก้หนี้ (บัตร / สินเชื่อ / นอกระบบ)",
        "tone_note": (
            "Debt CAN carry shame — but lead with empathy ONLY when the user "
            "actually voices the feeling (R12 words: เครียด/ไม่ไหว/กลัว/…). "
            "For a NEUTRAL ask ('แนะนำวิธีจัดการหนี้บัตร') do NOT open with "
            "'เข้าใจว่าหนี้กดดัน…' — lead with the numbers + plan, and put any "
            "warm/normalizing line as ONE short sentence at the END. NEVER lecture."
        ),
        "discovery_checklist": [
            "หนี้แบบไหนกดดันสุดตอนนี้ (เริ่มข้อเดียวก่อน — ไม่ยิง checklist รวด)",
            "หนี้แต่ละก้อน: ยอด / ดอก % / ขั้นต่ำต่อเดือน",
            "รายได้ net + รายจ่ายจำเป็นต่อเดือน",
            "มีหนี้นอกระบบมั้ย (ถามตอน user อุ่นแล้ว, sensitive สุด)",
            "มีทรัพย์ที่ใช้ refinance/consolidation ได้มั้ย (บ้าน / รถ)",
        ],
        "framework": {
            "avalanche": (
                "Pay minimum on all, throw extra at HIGHEST-rate debt first. "
                "Saves the most money. Use when user is disciplined and patient."
            ),
            "snowball": (
                "Pay minimum on all, throw extra at SMALLEST-balance debt first. "
                "Gives psychological wins. Use when user is discouraged or has "
                "failed payoff attempts before."
            ),
            "consolidation": (
                "Move high-rate debt (credit card 16-25%) into a lower-rate "
                "loan (home equity / P-loan from major bank, 7-12%). "
                "ONLY safe if user has the discipline NOT to re-borrow on the "
                "cleared cards. Otherwise it becomes 'consolidate then re-spend'."
            ),
            "thai_debt_clinic": (
                "Bank of Thailand's Debt Clinic (debtclinic.or.th) consolidates "
                "credit-card NPL across multiple lenders, can drop rate to 3-7%."
            ),
            "informal_debt": (
                "Interest > 15%/yr is illegal in Thailand (Civil Code §654). "
                "User can refuse to pay illegal rates and file with police."
            ),
            "sequence": [
                "Build 1-month emergency fund FIRST (tiny but vital)",
                "Pay minimums on ALL debts on time (avoid default + penalty APR)",
                "Choose avalanche or snowball, throw surplus there",
                "Close paid-off cards immediately (don't re-use)",
                "Track progress monthly to keep momentum",
            ],
        },
        "red_flags": [
            "Borrowing new debt to pay old debt (without consolidation plan)",
            "Skipping minimum payments to save 'for emergency'",
        ],
    },

    "tax": {
        "topic_label": "วางแผนภาษี / ลดหย่อน (Thailand)",
        "tone_note": (
            "Tax planning is high-stakes when amounts are large — append R8 "
            "disclaimer and suggest a tax advisor for complex situations "
            "(business income, foreign income, large transfers)."
        ),
        "discovery_checklist": [
            "ประเภทรายได้ (40(1) เงินเดือน / 40(2) ค่าจ้าง / 40(8) ธุรกิจ / อื่นๆ)",
            "รายได้รวมต่อปี (ประมาณ)",
            "ลดหย่อนพื้นฐานที่ใช้แล้ว (ประกันสังคม / ประกันสุขภาพ / บุตร / พ่อแม่)",
            "เคยซื้อ SSF / RMF / ประกันบำนาญ / ประกันชีวิตมั้ย",
            "ผ่อนบ้านมั้ย (ดอกเบี้ยลดหย่อนได้สูงสุด 100,000)",
        ],
        "framework": {
            "marginal_rate_thinking": (
                "Tax savings = deduction amount × marginal rate. "
                "At 25-35% bracket, deductions are powerful. "
                "At 5-10% bracket, locking up cash in RMF/SSF often "
                "isn't worth the liquidity loss."
            ),
            "deduction_priority": [
                "SSF — sellable after 10 years from purchase date",
                "RMF — must hold until age 55 AND ≥ 5 years",
                "ประกันบำนาญ — flexible but caps lower",
                "ประกันชีวิต — max 100k, only if you need the coverage anyway",
                "บริจาค — only when amount is meaningful and aligned with values",
            ],
            "caps_2026": (
                "SSF + RMF + ประกันบำนาญ + กบข./PVD = max 500,000 OR "
                "30% of taxable income, whichever is lower. "
                "Home loan interest: max 100,000."
            ),
            "common_mistake": (
                "Buying RMF/SSF at year-end to hit the cap, then needing "
                "the cash next year — locked in for 5-10 years minimum."
            ),
        },
        "red_flags": [
            "Buying tax shelters that lock up emergency fund",
            "Insurance bought for tax benefit but no actual coverage need",
            "Foreign income / crypto / business income — needs real advisor",
        ],
    },

    "invest": {
        "topic_label": "เริ่มลงทุน / asset allocation",
        "tone_note": (
            "Mint Money gives GENERAL perspective only. Never recommend "
            "specific stocks, crypto allocations, derivatives, or leveraged "
            "products. Always append R8 disclaimer."
        ),
        "prerequisite_check": [
            "Emergency fund of 6 months' expenses — if missing, FIX THIS FIRST",
            "High-interest debt (cards 16%+) — pay this off FIRST. "
            "Paying off a 16% card = 16% guaranteed return. "
            "No investment matches a guaranteed 16%.",
        ],
        "discovery_checklist": [
            "อายุ + timeline จนถึงเกษียณ",
            "ความเสี่ยงรับได้ — ถ้าพอร์ตลง 30% ใน 1 ปี จะนอนหลับมั้ย",
            "ประสบการณ์ลงทุน (มือใหม่ / มี / เคยขาดทุนหนัก)",
            "มีเป้าหมายเฉพาะมั้ย (เกษียณ / ดาวน์บ้าน / ทุนลูก)",
        ],
        "framework": {
            "asset_allocation_rule": (
                "Equities % ≈ 100 - age "
                "(age 30 → 70% equity / 30% bonds; age 50 → 50/50). "
                "Adjust by risk tolerance — conservative subtract 10."
            ),
            "beginner_path": (
                "Start with monthly DCA into broad-index funds "
                "(SET50, S&P 500 via FIF). Low fees, no stock-picking. "
                "Add bond fund for stability."
            ),
            "diversification_min": (
                "Across asset classes: equity + fixed income + cash. "
                "Geographic: Thai + foreign. Currency hedged if income is THB."
            ),
            "what_not_to_recommend": [
                "Specific stocks, IPOs",
                "Crypto allocation (highly volatile, unregulated guidance risk)",
                "Derivatives, options, leveraged ETFs",
                "Anything 'guaranteed high return' — defer to expert + warn",
            ],
        },
        "red_flags": [
            "Wanting to invest while carrying credit card debt",
            "No emergency fund yet",
            "Investing borrowed money",
            "'My friend made X% on Y' — anchoring on lucky outcome",
        ],
    },

    "discipline": {
        "topic_label": "วินัยการเงิน / budget / emergency fund",
        "tone_note": (
            "Discipline talks CAN follow shame ('I can't save') — but "
            "empathize-FIRST only if the user voices that feeling (R12). For a "
            "neutral ask, lead with the system/plan; if you normalize ('most "
            "people don't budget by default'), put it as a CLOSING line, not "
            "the opener. The system they need is automatic, not heroic willpower."
        ),
        "discovery_checklist": [
            "รายได้ net ต่อเดือน",
            "รายจ่ายคงที่ (ค่าเช่า/ผ่อน/บิลล์/ประกัน)",
            "รายจ่ายแปรผัน (กิน/เที่ยว/ช้อปปิ้ง — ประมาณก็พอ)",
            "เงินเก็บปัจจุบัน",
            "เป้าหมาย — ออมเพื่ออะไร, ภายในเมื่อไร",
        ],
        "framework": {
            "50_30_20": (
                "50% needs (rent, food, bills), 30% wants (entertainment, "
                "dining out), 20% saving + debt payoff. "
                "Adjust downward for high cost-of-living areas."
            ),
            "emergency_fund_size": (
                "3 months expenses — single, stable job. "
                "6 months — has dependents OR moderate income volatility. "
                "12 months — freelance / commission / sole earner with family."
            ),
            "emergency_fund_where": (
                "High-yield savings account or money market fund. "
                "Liquid (withdraw within days), zero risk to principal. "
                "NOT in stocks or RMF/SSF."
            ),
            "pay_yourself_first": (
                "Auto-transfer to savings on payday BEFORE spending. "
                "Removes willpower from the loop."
            ),
            "small_starts_count": (
                "2,000/month consistently beats '10,000 someday'. "
                "Start tiny, grow as habits stick."
            ),
        },
        "red_flags": [
            "Saving plan that requires zero entertainment (won't stick)",
            "User skipping emergency fund to chase higher returns",
            "Tracking every baht manually — sustainable only weeks",
        ],
    },
}


_VALID_TOPICS = tuple(PLAYBOOKS.keys())


@tool
async def get_advice_playbook(
    topic: Literal[
        "home", "refinance", "car", "debt", "tax", "invest", "discipline"
    ],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """Load decision framework + discovery checklist for a major-life
    finance topic.

    Call BEFORE giving specific advice on these decisions. The returned
    dict guides what to ask the user and how to evaluate the answer.

    Args:
      topic: One of "home", "refinance", "car", "debt", "tax", "invest",
             "discipline". Pick the closest match — multiple framings
             collapse into the same playbook (e.g. "เก็บเงิน" → "discipline",
             "หนี้นอกระบบ" → "debt").

    Returns: ToolMessage with JSON of {
      topic_label, tone_note, discovery_checklist, framework, red_flags,
      [prerequisite_check for invest]
    }
    """
    _emit_status(_STATUS_WORD)

    playbook = PLAYBOOKS.get(topic)
    if playbook is None:
        slog("get_advice_playbook", f"unknown topic: {topic!r}")
        return ToolMessage(
            content=json.dumps({
                "error": f"unknown topic {topic!r}",
                "valid_topics": list(_VALID_TOPICS),
            }),
            tool_call_id=tool_call_id,
        )

    slog("get_advice_playbook", f"loaded topic={topic}")
    return ToolMessage(
        content=json.dumps({"topic": topic, **playbook}, ensure_ascii=False),
        tool_call_id=tool_call_id,
    )


def _emit_status(word: str) -> None:
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"status": word})
    except Exception:
        pass


__all__ = ["get_advice_playbook", "PLAYBOOKS"]
