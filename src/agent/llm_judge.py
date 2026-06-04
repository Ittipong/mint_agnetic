"""LLM-as-judge — grade real-mode agent answers against per-scenario rubrics.

Used by real-mode Playwright tests (marked @pytest.mark.real_llm). The
judge is a separate LLM call to a strong reasoning model (configurable
via JUDGE_MODEL env, defaults to anthropic/claude-3.5-sonnet). It returns
a structured {passed: bool, reason: str} verdict.

Why a separate LLM, not the agent's own model: avoid self-grading bias.
The judge sees only the user question + agent answer + rubric — not the
agent's prompts or trace.
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

import httpx

from .llm_openrouter import _resolve_provider


class Verdict(TypedDict):
    passed: bool
    reason: str


JUDGE_SYSTEM = """You grade financial chatbot answers. You are STRICT but FAIR.

Given:
- user_question (Thai)
- agent_answer (Thai or mixed)
- rubric (English) — what a correct answer must satisfy

Reply JSON ONLY:
{"passed": true|false, "reason": "<one short sentence>"}

Rules:
1. If the rubric says "must mention a specific number" and the answer has no number → fail.
2. If the rubric says "must answer in Thai" and the answer is mostly English → fail.
3. Hallucination check: if the agent answer asserts a number, that number must be plausible per the rubric.
4. Paraphrase is fine — exact wording is NOT required.
5. Be concise. One sentence reason."""


async def judge(
    user_question: str,
    agent_answer: str,
    rubric: str,
    *,
    model: str | None = None,
) -> Verdict:
    """Call the judge LLM and return verdict.

    Raises RuntimeError if the judge response is unparseable — caller
    treats this as test infrastructure failure, not an agent failure.
    """
    model = model or os.getenv("JUDGE_MODEL", "deepseek/deepseek-v4-pro")
    api_key = os.environ["OPENROUTER_API_KEY"]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"user_question: {user_question}\n"
                    f"agent_answer: {agent_answer}\n"
                    f"rubric: {rubric}\n"
                    "Grade now."
                ),
            },
        ],
        "temperature": 0,
    }
    # Apply the same provider routing prefs (sort=latency / order) as the
    # runtime LLM calls so no chat/completions request bypasses it.
    provider = _resolve_provider()
    if provider is not None:
        payload["provider"] = provider

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fence if the judge wrapped its JSON in a code block.
    if text.startswith("```"):
        text = (
            text.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

    try:
        data = json.loads(text)
        return {"passed": bool(data["passed"]), "reason": str(data["reason"])}
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"judge returned unparseable JSON: {text!r}") from exc
