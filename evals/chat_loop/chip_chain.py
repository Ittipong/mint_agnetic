#!/usr/bin/env python3
"""Chip-chain QA: tap the follow-up suggestion chips N times in a row.

Simulates a user who asks one question and then only taps the suggested
next question, over and over, in ONE thread — through the Cloudflare tunnel
(the same path as the phone). The UI is too slow for this, so it runs at the
API level.

Usage: chip_chain.py [--hops 6] [--pick rotate|first|last] [--only ID ...]
Writes results_chain/<id>.json: per turn the question, chips offered, chip
picked, answer, and timings (first/last answer token, chips, done).

Correctness is NOT judged here — verify each answer's numbers against the
DB afterwards (see docs/qa_chip_chain_*.md).
"""
import argparse, json, os, sys, time, uuid, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://chat.minttechdev.uk"
USER = os.environ.get("QA_USER", "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_chain")

SEEDS = {
    "C1_spend": "เดือนนี้ใช้จ่ายไปเท่าไหร่",
    "C2_balance": "ตอนนี้เหลือเงินเท่าไหร่",
    "C3_card": "บัตรเครดิตต้องจ่ายเท่าไหร่ เมื่อไหร่",
    "C4_advice": "อยากเก็บเงินให้ได้มากขึ้น ควรเริ่มยังไง",
}


def turn(thread, msg, timeout=150):
    """One SSE turn; returns answer, blocks and wall-clock event timings."""
    body = json.dumps({"user_id": USER, "thread_id": thread, "message": msg}).encode()
    req = urllib.request.Request(BASE + "/chat/stream", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream",
                                          "User-Agent": "curl/8.4.0"})
    t0 = time.time()
    ev, data_lines, answer, blocks, errors = None, [], [], [], []
    tm = {}  # event → first-seen seconds; answer_last tracked separately

    def mark(key):
        tm.setdefault(key, round(time.time() - t0, 2))

    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:][1:] if line[5:].startswith(" ") else line[5:])
            elif line == "" and ev:
                data = "\n".join(data_lines)
                data_lines = []
                if ev == "answer_token":
                    mark("answer_first")
                    tm["answer_last"] = round(time.time() - t0, 2)
                    answer.append(data)
                elif ev == "block":
                    b = json.loads(data)
                    blocks.append(b)
                    if b.get("type") == "suggestions":
                        mark("chips")
                elif ev == "error":
                    errors.append(data)
                else:
                    mark(ev)
                ev = None
    tm["total"] = round(time.time() - t0, 2)
    chips = next((b.get("items") or [] for b in blocks if b.get("type") == "suggestions"), [])
    return dict(msg=msg, answer="".join(answer), chips=chips, errors=errors,
                block_types=[b.get("type") for b in blocks], timing=tm)


def pick_chip(chips, hop, how):
    if not chips:
        return None
    if how == "first":
        return chips[0]
    if how == "last":
        return chips[-1]
    return chips[hop % len(chips)]  # rotate: vary the axis the user follows


def run_chain(cid, seed, hops, how):
    thread = f"chip-{cid}-{uuid.uuid4().hex[:6]}"
    res = {"id": cid, "thread": thread, "pick": how, "turns": []}
    msg, label = seed, None
    for hop in range(hops + 1):  # seed turn + `hops` chip taps
        try:
            t = turn(thread, msg)
        except Exception as e:  # keep the transcript even on a network error
            t = dict(msg=msg, answer="", chips=[], errors=[repr(e)], block_types=[], timing={})
        t["hop"], t["chip_label"] = hop, label
        res["turns"].append(t)
        tmg = t["timing"]
        print(f"[{cid} hop{hop}] {msg[:50]!r} first={tmg.get('answer_first')} "
              f"last={tmg.get('answer_last')} chips={tmg.get('chips')} total={tmg.get('total')} "
              f"n_chips={len(t['chips'])}", flush=True)
        if hop == hops:
            break
        chip = pick_chip(t["chips"], hop, how)
        if chip is None:
            res["broken_at"] = hop  # the chain dead-ends: no chip to tap
            break
        msg = chip.get("send") or chip.get("label")
        label = chip.get("label")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{cid}_{how}.json"), "w"), ensure_ascii=False, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=6)
    ap.add_argument("--pick", default="rotate", choices=["rotate", "first", "last"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--parallel", type=int, default=2)
    a = ap.parse_args()
    seeds = {k: v for k, v in SEEDS.items() if not a.only or k in a.only}
    with ThreadPoolExecutor(a.parallel) as pool:
        results = list(pool.map(lambda kv: run_chain(kv[0], kv[1], a.hops, a.pick), seeds.items()))
    tone_report(results)


def tone_report(results):
    """Count answers that open by ascribing a feeling the user never voiced
    (R12 — docs/qa_chip_chain_2026-09-27.md)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from src.agent.streaming.tone_guard import _PROJECTED_FEELING, user_voiced_feeling

    turns = [t for r in results for t in r["turns"]]
    bad = [t for t in turns
           if _PROJECTED_FEELING.search(t["answer"].lstrip().split("\n\n", 1)[0])
           and not user_voiced_feeling(t["msg"])]
    print(f"tone: {len(bad)}/{len(turns)} answers open with an unasked feeling", flush=True)
    for t in bad:
        print(f"  - {t['msg'][:50]!r} → {t['answer'][:80]!r}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
