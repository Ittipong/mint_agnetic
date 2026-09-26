"""Assert ground-truth facts per scenario (dev DB as of 2026-09-26 evening,
including another session's "ค่าขนมบ่าย 45" row on Sep 26). turn index -> checks.
need: substrings that must appear; forbid: must not appear;
prop: expected proposal fields; noprop: no proposal block expected."""
import json, glob, sys
E = {
 "S01_balance": {0: {"need": ["153,059", "24,943", "603", "42,000", "60,000"]}},
 "S02_spend_month": {0: {"need": ["44,334"]}},
 "S03_spend_last": {0: {"need": ["32,140"]}},
 "S04_income": {0: {"need": ["42,000"]}},
 "S06_coffee": {0: {"need": ["1,080"]}},
 "S07_card": {0: {"need": ["24,111", "5 ต"]}},
 "S08_goal": {0: {"need": ["42,000", "60,000", "18,000"]}},
 "S09_budget": {0: {"need": ["7,000", "255"]}},
 "S10_avg6m": {0: {"need": ["29,756"]}},
 "S12_subs": {0: {"need": ["603"]}},
 "S13_trip": {0: {"need": ["9,350"]}},
 "S14_add_coffee": {0: {"prop": {"amount": 65, "category": "กาแฟ"}}},
 "S15_add_income": {0: {"prop": {"amount": 42000, "type": "income"}}},
 "S16_add_wallet": {0: {"prop": {"amount": 150, "wallet_id": "08f38d01-cf82-5b69-9971-70d0f12eb7ef"}}},
 "S17_add_yesterday": {0: {"prop": {"amount": 120, "date": "2026-09-25"}}},
 "S18_add_card": {0: {"prop": {"amount": 890, "wallet_id": "54bdd26a-2348-55ec-821f-177bdd673dfa"}}},
 "S19_add_noamount_cat": {0: {"noprop": True}},
 "S20_emergency": {0: {"need": ["37,005"], "forbid": ["43,708"]}},
 "S21_afford": {0: {"need": ["17,161"], "forbid": ["68,167", "43,708", "24,458"]}},
 "S23_halluc": {0: {"need": ["44,334", "ไม่ใช่"]}},
 "S25_offtopic": {0: {"forbid": ["```", "sorted("]}},
 "S26_cash_spend": {0: {"need": ["1,387"]}},
 "S27_transport_aug": {0: {"need": ["8,442"]}},
 "S28_bonus": {0: {"need": ["30,000"]}},
 "S29_multi_add": {0: {"group": {"rows": 3, "total": 184}}},
 "S39_add_transfer": {0: {"noprop": True}},
 "S42_add_date": {0: {"prop": {"amount": 9500, "date": "2026-09-01"}}},
 "S49_forecast": {0: {"forbid": ["ซุปเปอร์มาร์เก็ต** — 4,024"]}},
 "M01_followup_sum": {0: {"need": ["44,334"]}, 1: {"need": ["32,140"]}, 2: {"need": ["76,474"]}},
 "M02_correction": {0: {"prop": {"amount": 80}}, 1: {"prop": {"amount": 90}}},
 "M03_clarify": {0: {"noprop": True}, 1: {"prop": {"amount": 250}}},
 "M04_drill": {0: {"need": ["ช้อปปิ้ง", "45,800", "29,613"]}},
 "S05_topcat": {0: {"need": ["ช้อปปิ้ง", "25,345", "9,912"]}},
 "S47_card_due": {0: {"need": ["5 ต", "24,111"]}},
 "M06_card_advice": {0: {"need": ["24,111"]}},
 "M05_goal_plan": {0: {"need": ["18,000"]}},
 "M07_eq_then_data": {1: {"need": ["54,167", "37,005"], "forbid": ["68,167"]}},
 "M08_add_then_ask": {0: {"prop": {"amount": 1200}}, 1: {"need": ["1,278"]}},
 "M09_topic_switch": {0: {"need": ["1,080"]}, 1: {"prop": {"amount": 55}}},
 "M14_multi_continue": {0: {"group": {"rows": 2, "total": 140}}},
}
fails = 0; checked = 0
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    r = json.load(open(f)); exp = E.get(r["id"])
    if not exp: continue
    for ti, c in exp.items():
        checked += 1
        if ti >= len(r["turns"]): print(f"FAIL {r['id']}[{ti}] missing turn"); fails += 1; continue
        t = r["turns"][ti]; a = t["answer"]; errs = []
        if t["errors"]: errs.append(f"error {t['errors'][0][:80]}")
        for n in c.get("need", []):
            if n not in a: errs.append(f"missing {n!r}")
        for n in c.get("forbid", []):
            if n in a: errs.append(f"forbidden {n!r}")
        props = [b for b in t["blocks"] if b.get("type") == "transaction_proposal"]
        if c.get("noprop") and props: errs.append("unexpected proposal")
        if "prop" in c:
            if not props: errs.append("no proposal")
            else:
                tx = props[-1]["transaction"]
                for k, v in c["prop"].items():
                    if tx.get(k) != v: errs.append(f"prop.{k}={tx.get(k)!r} != {v!r}")
        if "group" in c:
            gs = [b for b in t["blocks"] if b.get("type") == "transaction_proposal_group"]
            if not gs:
                errs.append("no group card")
            elif (len(gs[-1]["transactions"]), gs[-1]["total"]) != (c["group"]["rows"], c["group"]["total"]):
                errs.append(f"group={len(gs[-1]['transactions'])} rows/{gs[-1]['total']} != {c['group']}")
        if c.get("noblock") and any(b.get("type") == c["noblock"] for b in t["blocks"]):
            errs.append(f"unexpected {c['noblock']}")
        if errs: fails += 1; print(f"FAIL {r['id']}[{ti}]: " + "; ".join(errs))
print(f"\n{checked - fails}/{checked} checks passed")
