"""Tolerant JSON extractor for noisy LLM output.

Ported from v2 (`nodes/_safe_json.py`) in Wave 1 — relocated under
`src/agent/utils/` because v3 has no `nodes/` package (the v2 node-graph
was replaced by Pure ReAct + standalone tools).
"""

from __future__ import annotations

import json
from typing import Any


def safe_json(text: str) -> dict[str, Any]:
    """Extract one JSON object from `text` tolerating common LLM noise.

    Strategy:
    1. Strip ```json fences.
    2. Try `json.loads` directly.
    3. Otherwise, find the first balanced `{...}` block (string-aware) and parse it.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = (
            cleaned.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in LLM output: {text[:80]!r}")

    depth, in_str, esc = 0, False, False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start : i + 1])
    raise ValueError(f"unbalanced JSON in LLM output: {text[:80]!r}")
