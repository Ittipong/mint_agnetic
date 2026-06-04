"""Compare flat ReAct vs create_react_agent over the live tunnel.

Runs a fixed prompt battery against https://chat.minttechdev.uk/chat/stream,
parses the SSE, and writes a normalized per-turn SIGNATURE to a JSON file.
Run it once per impl (flip REACT_IMPL + reload between runs), then diff the two
signature files with `--diff a.json b.json`.

A signature captures only what MUST match for parity (data, not phrasing):
  - block types emitted (sorted set)
  - transaction_proposal present?  (ADD card)
  - suggestions present? + chip count
  - numbers in the final answer text (set)  ← the load-bearing comparison
  - soft-warn footer present?
  - answer_token / status_token counts (informational; flat may stream more)

Usage:
  python scripts/compare_react_impl.py --run --label flat   --out /tmp/sig_flat.json
  python scripts/compare_react_impl.py --run --label prebuilt --out /tmp/sig_prebuilt.json
  python scripts/compare_react_impl.py --diff /tmp/sig_prebuilt.json /tmp/sig_flat.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

URL = "https://chat.minttechdev.uk/chat/stream"
USER = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
WALLET = "17f0a617-87f2-44a5-8c7d-8eac1fbe4839"

# (id, [turns]) — multi-turn threads share state. Covers every mode.
BATTERY = [
    ("balance", ["เหลือเงินเท่าไหร่"]),
    ("analyst_month", ["เดือนนี้ใช้ไปเท่าไหร่"]),
    ("analyst_bycat", ["เดือนนี้ใช้หมวดไหนเยอะสุด"]),
    ("advisor_car", ["ฉันอยากซื้อรถ Honda city"]),
    ("advisor_debt", ["หนี้บัตรเครดิตเยอะ ควรทำยังไงดี"]),
    ("add_simple", ["เพิ่ม 60 กาแฟ"]),
    ("add_income", ["ได้เงินเดือน 32000"]),
    ("greeting", ["สวัสดีครับ"]),
    ("card_debt", ["บัตรเครดิตใช้ไปเท่าไหร่"]),
    ("goal", ["เป้าหมายเที่ยวอังกฤษเหลือเท่าไหร่"]),
    ("followup_total", ["ยอดแต่ละกระเป๋าเท่าไหร่", "รวมทั้งหมดเท่าไหร่"]),
]

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_FOOTER = "⚠️ หมายเหตุ"


def _nums(text: str) -> list[str]:
    """Numbers in answer text, comma-stripped + deduped (the parity load)."""
    return sorted({m.group(0).replace(",", "") for m in _NUM.finditer(text or "")})


def _curl(thread_id: str, message: str) -> str:
    payload = json.dumps(
        {"user_id": USER, "thread_id": thread_id, "wallet_id": WALLET, "message": message},
        ensure_ascii=False,
    )
    r = subprocess.run(
        ["curl", "-s", "-N", "-X", "POST", URL, "-H", "Content-Type: application/json",
         "-d", payload, "--max-time", "120"],
        capture_output=True, text=True,
    )
    return r.stdout


def _sig_from_sse(raw: str) -> dict:
    answer_tokens = raw.count("event: answer_token")
    status_tokens = raw.count("event: status_token")
    block_types, answer_text, chip_count = set(), "", 0
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[len("data: "):])
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        t = obj.get("type")
        if t:
            block_types.add(t)
        if t == "answer":
            answer_text = obj.get("text", "")
        if t == "suggestions":
            chip_count = len(obj.get("items", []))
    return {
        "block_types": sorted(block_types),
        "has_proposal": "transaction_proposal" in block_types
        or "transaction_proposal_group" in block_types,
        "has_suggestions": "suggestions" in block_types,
        "chip_count": chip_count,
        "answer_numbers": _nums(answer_text),
        "soft_warn": _FOOTER in answer_text,
        "answer_tokens": answer_tokens,
        "status_tokens": status_tokens,
        "answer_preview": (answer_text or "")[:80],
    }


def run(label: str, out: str) -> None:
    import time
    sigs = {}
    for cid, turns in BATTERY:
        tid = f"cmp-{label}-{cid}-{int(time.time()*1000)}"
        last = ""
        for msg in turns:
            last = _curl(tid, msg)
        sigs[cid] = _sig_from_sse(last)
        s = sigs[cid]
        print(f"[{label}] {cid:16} blocks={s['block_types']} "
              f"nums={s['answer_numbers']} proposal={s['has_proposal']} "
              f"chips={s['chip_count']} softwarn={s['soft_warn']}")
    with open(out, "w") as f:
        json.dump(sigs, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


# Fields that MUST match for parity. answer_tokens/status_tokens excluded
# (flat may stream differently — that's "better or equal", not a regression).
_PARITY_FIELDS = ["block_types", "has_proposal", "has_suggestions",
                  "answer_numbers", "soft_warn"]


def diff(path_a: str, path_b: str) -> None:
    a = json.load(open(path_a)); b = json.load(open(path_b))
    name_a, name_b = path_a, path_b
    print(f"PARITY DIFF  A={name_a}  B={name_b}\n" + "=" * 60)
    all_ok = True
    for cid in a:
        if cid not in b:
            print(f"  {cid}: MISSING in B"); all_ok = False; continue
        diffs = []
        for fld in _PARITY_FIELDS:
            if a[cid].get(fld) != b[cid].get(fld):
                diffs.append(f"{fld}: {a[cid].get(fld)} != {b[cid].get(fld)}")
        if diffs:
            all_ok = False
            print(f"  ❌ {cid}")
            for d in diffs:
                print(f"       {d}")
        else:
            print(f"  ✅ {cid}")
    print("=" * 60)
    print("RESULT:", "PARITY ✅ (flat ≡ prebuilt on all parity fields)" if all_ok
          else "DIVERGENCE ❌ — see above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="/tmp/sig.json")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.diff:
        diff(args.diff[0], args.diff[1])
    elif args.run:
        run(args.label, args.out)
    else:
        ap.print_help()
