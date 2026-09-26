#!/usr/bin/env python3
"""Chat QA harness: run scenarios (single / multi-turn) against the tunnel.

Usage: chat_qa.py scenarios.json [only_id ...]
scenario = {"id": "...", "turns": ["msg1", "msg2", ...]}
Writes results/<id>.json and prints a compact transcript.
"""
import json, os, subprocess, sys, time, uuid, urllib.request

BASE = "https://chat.minttechdev.uk"
_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "backend", ".env.dev")
DB_URL = next((l.split("=", 1)[1].strip() for l in open(_ENV) if l.startswith("DATABASE_URL=")), "")
WRITTEN: list[str] = []  # sync_ids this run inserted — printed at the end for cleanup
USER = os.environ.get("QA_USER", "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9")
WALLET = os.environ.get("QA_WALLET")  # the chat-input wallet pick, as the app sends it
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def turn(thread, msg, timeout=150):
    body = json.dumps({"user_id": USER, "thread_id": thread, "message": msg,
                       **({"wallet_id": WALLET} if WALLET else {})}).encode()
    req = urllib.request.Request(BASE + "/chat/stream", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream",
                                          "User-Agent": "curl/8.4.0"})
    t0 = time.time()
    ev, answer, blocks, status, errors, other = None, [], [], [], [], []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data_lines = []
        for raw in r:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:][1:] if line[5:].startswith(" ") else line[5:])
            elif line == "":
                data = "\n".join(data_lines)
                data_lines = []
                if ev == "answer_token":
                    answer.append(data)
                elif ev == "block":
                    try:
                        blocks.append(json.loads(data))
                    except Exception:
                        blocks.append({"raw": data})
                elif ev == "status_token":
                    status.append(data)
                elif ev == "error":
                    errors.append(data)
                elif ev:
                    other.append(ev)
                ev = None
    return dict(msg=msg, answer="".join(answer), blocks=blocks, status=status,
                errors=errors, other=other, secs=round(time.time() - t0, 1))


def confirm(thread, proposal_id):
    body = json.dumps({"user_id": USER, "thread_id": thread,
                       "proposal_id": proposal_id}).encode()
    req = urllib.request.Request(BASE + "/transactions/confirm", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "curl/8.4.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def write_and_confirm(thread, proposal):
    """What tapping ยืนยัน does in the app/prototype: write the ledger row
    (the agent never writes transactions), then mark the proposal confirmed."""
    # A group card (multi-item text / slip) saves every row, then ONE confirm.
    # Rows go straight into the dev DB with the same columns insight's
    # /dev/api/transaction writes (that endpoint is now key- and
    # allowlist-gated, and test users are not on the list).
    txs = proposal.get("transactions") or [proposal["transaction"]]
    wrote = []
    for tx in txs:
        sync_id = tx.get("sync_id") or str(uuid.uuid4())
        eff = 1 if tx["type"] == "income" else -1
        def q(v):
            return "NULL" if v is None else "'" + str(v).replace("'", "''") + "'"
        sql = (
            "INSERT INTO transactions (sync_id, created_by_user_id, type, amount, date, note, "
            "currency_code, currency_symbol, category_sync_id, wallet_sync_id, effect_on_wallet, "
            "icon, category_name, status, include_in_report, version, is_deleted, is_recurring, "
            "created_at, updated_at) "
            f"SELECT {q(sync_id)}, {q(USER)}, {q(tx['type'])}, {float(tx['amount'])}, "
            f"COALESCE({q(tx.get('date'))}::timestamptz, now()), {q(tx.get('note'))}, 'THB', '฿', "
            f"{q(tx.get('category_sync_id'))}, {q(tx.get('wallet_sync_id'))}, {eff}, "
            "c.icon, c.name, 'confirmed', true, 1, false, false, now(), now() "
            f"FROM (SELECT 1) one LEFT JOIN categories c ON c.sync_id = {q(tx.get('category_sync_id'))} "
            "ON CONFLICT (sync_id) DO NOTHING;"
        )
        subprocess.run(["psql", DB_URL, "-v", "ON_ERROR_STOP=1", "-qtAc", sql], check=True,
                       capture_output=True, text=True)
        WRITTEN.append(sync_id)
        wrote.append({"sync_id": sync_id, "amount": tx["amount"]})
    return wrote, confirm(thread, proposal["proposal_id"])


def main():
    scen = json.load(open(sys.argv[1]))
    only = set(sys.argv[2:])
    for s in scen:
        if only and s["id"] not in only:
            continue
        thread = f"qa-{s['id']}-{uuid.uuid4().hex[:6]}"
        res = {"id": s["id"], "thread": thread, "turns": []}
        print(f"\n######## {s['id']}  thread={thread}")
        last_prop = None
        last_block = None
        for m in s["turns"]:
            if m == "__CONFIRM_WRITE__":
                out = write_and_confirm(thread, last_block) if last_block else "no proposal"
                print(f"\n=== CONFIRM+WRITE: {str(out)[:160]}")
                res["turns"].append(dict(msg=m, answer=str(out), blocks=[], status=[], errors=[], other=[], secs=0))
                continue
            if m == "__CONFIRM__":
                out = confirm(thread, last_prop) if last_prop else "no proposal"
                print(f"\n=== CONFIRM {last_prop}: {out[:120]}")
                res["turns"].append(dict(msg=m, answer=out, blocks=[], status=[], errors=[], other=[], secs=0))
                continue
            try:
                t = turn(thread, m)
            except Exception as e:  # keep going on network errors
                t = dict(msg=m, answer="", blocks=[], status=[], errors=[repr(e)], other=[], secs=0)
            res["turns"].append(t)
            for b in t["blocks"]:
                if b.get("type") in ("transaction_proposal", "transaction_proposal_group"):
                    last_prop = b.get("proposal_id")
                    last_block = b
            print(f"\n>>> USER: {m}   ({t['secs']}s)")
            if t["errors"]:
                print("!!! ERROR:", t["errors"])
            for b in t["blocks"]:
                bt = b.get("type")
                if bt in ("suggestions",):
                    print("  [chips]", [c.get("label", c) if isinstance(c, dict) else c
                                        for c in (b.get("items") or b.get("suggestions") or [])])
                else:
                    print("  [block]", json.dumps(b, ensure_ascii=False)[:700])
            print("<<< BOT:", t["answer"][:2500])
        json.dump(res, open(os.path.join(OUT, f"{s['id']}.json"), "w"), ensure_ascii=False, indent=1)
    if WRITTEN:
        with open(os.path.join(OUT, "written_sync_ids.txt"), "a") as f:
            f.write("\n".join(WRITTEN) + "\n")
        print(f"\nWROTE {len(WRITTEN)} transaction(s) → results/written_sync_ids.txt (delete after the run)")


if __name__ == "__main__":
    main()
