#!/usr/bin/env python3
"""Numeric-correctness baseline eval — quick smoke for v3 advisor numbers.

ADAPTED from v2 `evals/numeric_eval.py`. Same churn-robust bracket-read
pattern (compute GT BEFORE and AFTER the agent call; accept either), same
canonical filter set (`type, is_deleted=false, status='confirmed',
include_in_report=true`) — so the numbers we grade against are the same
ones v2 graded against. The only differences are:

  - v3 has NO sync JSON `/chat`; we POST to `/chat/stream` (SSE) and
    coalesce every `answer_token` data field into `answer` text.
  - We accept an env override for the v3 server URL (so smoke tests can
    point at the Cloudflare tunnel per `feedback_test_via_cloudflared`).

This script is a deterministic smoke for the advisor path — it's NOT the
full eval gate. The full gate is `evals/run_eval.py` (LangSmith dataset
+ 3-gate scoring). Use this when you want a quick sanity check after a
code change: 7 questions, ~1 min, no LangSmith dependency.

Today is fixed at 2026-05-28 for these runs: "this month" = May 2026,
"last month" = April 2026.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
SERVER = os.getenv("V3_SERVER", "http://127.0.0.1:2026") + "/chat/stream"
USER_ID = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
WALLET_ID = "00000000-0001-4000-a000-000000000001"


def _dsn() -> str:
    """Resolve BACKEND_DATABASE_URL from env or .env. v3 reuses the same
    dev DSN as v2 (memory `reference_dev_db_write_access`)."""
    if v := os.getenv("BACKEND_DATABASE_URL"):
        return v
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("BACKEND_DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("BACKEND_DATABASE_URL not found in env or .env")


# Canonical filters shared by income/expense/breakdown metrics.
_F = ("is_deleted=false AND status='confirmed' AND include_in_report=true "
      "AND created_by_user_id=%(u)s")
# May 2026 ("this month") and April 2026 ("last month"), agent's UTC window.
_MAY = "date >= '2026-05-01' AND date < '2026-06-01'"
_APR = "date >= '2026-04-01' AND date < '2026-05-01'"


# Each question: SQL returns a single scalar (the number to verify); `label`
# is an extra substring that must also appear (e.g. the top category name).
QUESTIONS = [
    {"id": "Q1", "q": "เดือนนี้ใช้จ่ายไปเท่าไหร่", "label": None,
     "sql": f"SELECT COALESCE(SUM(amount::numeric),0) FROM transactions WHERE type='expense' AND {_F} AND {_MAY}"},
    {"id": "Q2", "q": "เดือนนี้มีรายรับเท่าไหร่", "label": None,
     "sql": f"SELECT COALESCE(SUM(amount::numeric),0) FROM transactions WHERE type='income' AND {_F} AND {_MAY}"},
    {"id": "Q3", "q": "เดือนนี้ใช้จ่ายหมวดไหนมากที่สุด", "label": "_top_cat",
     "sql": f"SELECT COALESCE(category_name,'(uncategorized)'), SUM(amount::numeric) FROM transactions WHERE type='expense' AND {_F} AND {_MAY} GROUP BY 1 ORDER BY 2 DESC LIMIT 1"},
    {"id": "Q4", "q": "รายการจ่ายที่แพงที่สุดเดือนนี้เท่าไหร่", "label": None,
     "sql": f"SELECT COALESCE(MAX(amount::numeric),0) FROM transactions WHERE type='expense' AND {_F} AND {_MAY}"},
    {"id": "Q5", "q": "เดือนที่แล้วใช้จ่ายไปเท่าไหร่", "label": None,
     "sql": f"SELECT COALESCE(SUM(amount::numeric),0) FROM transactions WHERE type='expense' AND {_F} AND {_APR}"},
    {"id": "Q6", "q": "เดือนที่แล้วมีรายการทั้งหมดกี่รายการ", "label": None,
     "sql": f"SELECT COUNT(*) FROM transactions WHERE is_deleted=false AND status='confirmed' AND created_by_user_id=%(u)s AND {_APR}"},
    {"id": "Q7", "q": "เดือนนี้ค่ากาแฟไปเท่าไหร่", "label": None,
     "sql": f"SELECT COALESCE(SUM(amount::numeric),0) FROM transactions WHERE type='expense' AND category_name='กาแฟ' AND {_F} AND {_MAY}"},
]


def gt(conn, item) -> tuple:
    with conn.cursor() as cur:
        cur.execute(item["sql"], {"u": USER_ID})
        row = cur.fetchone()
    return row  # (number,) or (category, amount)


def _coalesce_answer_from_sse(text: str) -> str:
    """Extract the final answer prose from a /chat/stream SSE body.

    v3 streams the answer as `answer_token` events; concatenate every `data:`
    line on `event: answer_token` chunks. The terminal `answer` block also
    contains the full prose — we'd accept that if answer_token were empty.
    """
    parts: list[str] = []
    answer_block_text = ""
    for chunk in text.replace("\r\n", "\n").split("\n\n"):
        lines = chunk.splitlines()
        event = ""
        data_parts: list[str] = []
        for line in lines:
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_parts.append(line[len("data:"):].lstrip())
        data = "".join(data_parts)
        if event == "answer_token" and data:
            parts.append(data)
        elif event == "block" and data:
            try:
                blk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if blk.get("type") == "answer":
                answer_block_text = blk.get("text") or answer_block_text
    return "".join(parts) or answer_block_text or "(no answer)"


def ask(message: str) -> str:
    body = json.dumps({"thread_id": str(uuid.uuid4()), "user_id": USER_ID,
                       "message": message, "wallet_id": WALLET_ID}).encode()
    req = urllib.request.Request(
        SERVER, data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read().decode("utf-8", errors="replace")
    return _coalesce_answer_from_sse(raw)


def _num_forms(n) -> list[str]:
    """Comma-formatted variants of a number to look for in the Thai answer."""
    i = int(round(float(n)))
    return list({f"{i:,}", str(i)})


def _check(item, before, after, answer) -> tuple[bool, str]:
    expected = []
    for row in (before, after):
        if item["label"] == "_top_cat":
            cat, amt = row[0], row[1]
            expected.append((cat, _num_forms(amt)))
        else:
            expected.append((None, _num_forms(row[0])))
    for cat, forms in expected:
        num_ok = any(f in answer for f in forms)
        cat_ok = (cat is None) or (cat in answer)
        if num_ok and cat_ok:
            return True, f"matched {cat or ''} {forms}".strip()
    gtdesc = " | ".join(
        (f"{r[0]}={_num_forms(r[1])}" if item["label"] == "_top_cat" else f"{_num_forms(r[0])}")
        for r in (before, after))
    return False, f"GT(before|after)= {gtdesc}"


def main() -> int:
    only = set(a for a in sys.argv[1:] if a.startswith("Q"))
    conn = psycopg.connect(_dsn())
    passed = failed = 0
    for item in QUESTIONS:
        if only and item["id"] not in only:
            continue
        before = gt(conn, item)
        try:
            answer = ask(item["q"])
        except Exception as e:  # noqa: BLE001
            answer = f"(ERROR: {type(e).__name__}: {e})"
        after = gt(conn, item)
        conn.commit()  # release snapshot so next read sees latest
        ok, why = _check(item, before, after, answer)
        passed += ok
        failed += not ok
        print(f"\n{'PASS' if ok else 'FAIL'} {item['id']}: {item['q']}")
        print(f"   {why}")
        print(f"   answer: {answer}")
    conn.close()
    print(f"\n{'='*60}\nRESULT: {passed} passed, {failed} failed "
          f"({passed}/{passed+failed})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
