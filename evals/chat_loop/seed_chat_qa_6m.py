#!/usr/bin/env python3
"""
Realistic 6-month finance fixture for chat (mint_agentic) QA.

Why this exists (vs mock_transactions.py): the chat agent needs enough
history with clear, checkable patterns (daily coffee, rent, subscriptions,
card statements, a bonus month, a spending spike) to exercise analyst /
advisor questions — and every category must use the SAME system categories +
asset icons the mobile app seeds (read straight from category_seed_data.dart
so the two never drift).

Persona: office worker in Bangkok, salary 42,000 THB on the 25th.
Window: first day of (today - 5 months) .. today. No future-dated rows.

WARNING: --apply WIPES wallets / transactions / categories / budgets / tags /
recurring / obligation data for ALL users, then seeds one user.

Usage:
  python3 evals/chat_loop/seed_chat_qa_6m.py              # print SQL only
  python3 evals/chat_loop/seed_chat_qa_6m.py --apply      # wipe + seed dev DB
  python3 evals/chat_loop/seed_chat_qa_6m.py --apply --email someone@example.com
"""

import argparse
import calendar
import json
import os
import random
import re
import subprocess
import sys
import uuid
from datetime import date, timedelta

# mint_money monorepo root (mobile/ and backend/ are siblings of mint_agentic/).
# Lives in mint_agentic/evals/chat_loop/, so three levels up; override with
# MINT_MONEY_ROOT if the checkout layout differs.
ROOT = os.environ.get("MINT_MONEY_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SEED_DART = os.path.join(
    ROOT, "mobile/lib/data/datasources/local/seeders/category_seed_data.dart")
COLORS_DART = os.path.join(ROOT, "mobile/lib/theme/app_colors.dart")
ASSET_DIR = os.path.join(ROOT, "mobile")

NS = uuid.UUID("6d1f3a52-5c0e-4c8e-9a61-3f4b2d7c9e10")
rng = random.Random(20260926)
USER_ID = None  # resolved at runtime
TODAY = date.today()


def sid(key):
    """Deterministic sync_id so re-seeding yields the same ids."""
    return str(uuid.uuid5(NS, key))


def esc(s):
    if s is None:
        return "NULL"
    return "'" + str(s).replace("'", "''") + "'"


def add_months(d, n):
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


START = add_months(TODAY.replace(day=1), -5)


def icon_json(path, bg):
    return json.dumps({"type": "asset", "value": path, "bg": bg},
                      separators=(",", ":"))


# ─────────────────────────── categories (from mobile) ───────────────────────

def load_colors():
    txt = open(COLORS_DART, encoding="utf-8").read()
    return {m.group(1): m.group(2) for m in re.finditer(
        r"static const Color (\w+) = Color\((0x[0-9A-Fa-f]{8})\)", txt)}


def load_seed_categories():
    """Parse CategorySeedData(...) blocks per list in category_seed_data.dart."""
    txt = open(SEED_DART, encoding="utf-8").read()
    colors = load_colors()
    out = {}
    for list_name in ("_generalCategories", "_creditCardCategories"):
        start = txt.index(f"static const List<CategorySeedData> {list_name}")
        end = txt.index("];", start)
        body = txt[start:end]
        cats = []
        for blk in re.findall(r"CategorySeedData\((.*?)\n    \)", body, re.S):
            f = dict(re.findall(r"(\w+): '([^']*)'", blk))
            f.update(re.findall(r"(\w+): (\d+),", blk))
            bg = re.search(r"backgroundColor: AppColors\.(\w+)", blk).group(1)
            f["bg"] = colors[bg].upper().replace("0X", "0x")
            cats.append(f)
        out[list_name] = cats
    return out


# ─────────────────────────── wallets ────────────────────────────────────────

GENERAL = {
    "kbank": dict(name="KBank ออมทรัพย์", initial=68000, cat="savings",
                  icon=icon_json("assets/cetegory_icons/building.png", "0xFFDAF7E0")),
    "cash": dict(name="เงินสด", initial=1500, cat="cash",
                 icon=icon_json("assets/cetegory_icons/wallet.png", "0xFFF9E6AE")),
    "truemoney": dict(name="TrueMoney", initial=600, cat="eWallet",
                      icon=icon_json("assets/cetegory_icons/income.png", "0xFFD1E6FF")),
}
CARDS = {
    # initial_used = the pre-window statement, paid on the first due date.
    "ktc": dict(name="บัตร KTC", limit=80000, initial_used=7850, bill=20, due=5,
                icon=icon_json("assets/cetegory_icons/credit.png", "0xFFE5867A")),
    "scb": dict(name="บัตร SCB M", limit=50000, initial_used=603, bill=28, due=15,
                icon=icon_json("assets/cetegory_icons/credit.png", "0xFFE4E0FF")),
}
GOALS = {
    "japan": dict(name="เที่ยวญี่ปุ่น", initial=12000, target=60000,
                  target_date=date(2027, 3, 31),
                  icon=icon_json("assets/cetegory_icons/travel.png", "0xFFD1E6FF")),
    "emergency": dict(name="เงินสำรองฉุกเฉิน", initial=45000, target=150000,
                      target_date=None,
                      icon=icon_json("assets/cetegory_icons/piggy.png", "0xFFDAF7E0")),
}
ALL_WALLETS = {**{k: ("general", v) for k, v in GENERAL.items()},
               **{k: ("creditcard", v) for k, v in CARDS.items()},
               **{k: ("goal", v) for k, v in GOALS.items()}}

CAT = {}  # (wallet_key, system_key) -> dict(sync_id, name, type, icon)


def build_categories(seed):
    rows = []
    for wkey, (kind, _) in ALL_WALLETS.items():
        if kind == "goal":
            continue
        lst = seed["_generalCategories" if kind == "general" else "_creditCardCategories"]
        wsid = sid(f"wallet:{wkey}")
        for c in lst:
            csid = str(uuid.uuid5(uuid.UUID(wsid), c["systemKey"]))
            icon = icon_json(c["iconPath"], c["bg"])
            if not os.path.exists(os.path.join(ASSET_DIR, c["iconPath"])):
                raise SystemExit(f"missing asset {c['iconPath']}")
            parent = c.get("parentSystemKey")
            psid = str(uuid.uuid5(uuid.UUID(wsid), parent)) if parent else None
            CAT[(wkey, c["systemKey"])] = dict(sync_id=csid, name=c["name"],
                                               type=c["type"], icon=icon)
            rows.append(
                "INSERT INTO categories (sync_id, user_id, name, type, icon, display_order, "
                "parent_sync_id, wallet_sync_id, source, system_key, is_active, is_deleted, "
                "created_at, updated_at) VALUES ("
                f"{esc(csid)}, {esc(USER_ID)}, {esc(c['name'])}, {esc(c['type'])}, "
                f"{esc(icon)}::jsonb, {c.get('displayOrder', 0)}, {esc(psid)}, {esc(wsid)}, "
                f"'system', {esc(c['systemKey'])}, true, false, NOW(), NOW());")
    return rows


def build_wallets():
    rows = []
    for k, w in GENERAL.items():
        rows.append(
            "INSERT INTO general_wallets (sync_id, user_id, name, initial_balance, currency, icon, "
            f"wallet_category, created_at, updated_at) VALUES ({esc(sid('wallet:'+k))}, {esc(USER_ID)}, "
            f"{esc(w['name'])}, {w['initial']}, 'THB', {esc(w['icon'])}, {esc(w['cat'])}, "
            f"{esc(START.isoformat())}, NOW());")
    for k, w in CARDS.items():
        rows.append(
            "INSERT INTO creditcard_wallets (sync_id, user_id, name, credit_limit, initial_used, "
            "billing_cycle_day, payment_due_day, currency, icon, created_at, updated_at) VALUES ("
            f"{esc(sid('wallet:'+k))}, {esc(USER_ID)}, {esc(w['name'])}, {w['limit']}, {w['initial_used']}, "
            f"{w['bill']}, {w['due']}, 'THB', {esc(w['icon'])}, {esc(START.isoformat())}, NOW());")
    for k, w in GOALS.items():
        td = esc(w["target_date"].isoformat()) if w["target_date"] else "NULL"
        rows.append(
            "INSERT INTO goal_wallets (sync_id, user_id, name, initial_balance, currency, icon, "
            "target_amount, target_date, start_date, is_deleted, created_at, updated_at) VALUES ("
            f"{esc(sid('wallet:'+k))}, {esc(USER_ID)}, {esc(w['name'])}, {w['initial']}, 'THB', "
            f"{esc(w['icon'])}, {w['target']}, {td}, {esc(START.isoformat())}, false, "
            f"{esc(START.isoformat())}, NOW());")
    # Owner membership rows: sync pull visibility joins through wallet_members.
    for k, (kind, _) in ALL_WALLETS.items():
        table = {"general": "general_wallets", "creditcard": "creditcard_wallets",
                 "goal": "goal_wallets"}[kind]
        ws = sid("wallet:" + k)
        rows.append(
            "INSERT INTO wallet_members (sync_id, wallet_id, wallet_sync_id, wallet_type, user_id, "
            "is_owner, is_active, is_deleted, joined_at) "
            f"SELECT {esc('owner_'+ws)}, id, {esc(ws)}, {esc(kind)}, {esc(USER_ID)}, true, true, false, NOW() "
            f"FROM {table} WHERE sync_id = {esc(ws)};")
    return rows


# ─────────────────────────── transactions ───────────────────────────────────

TXNS = []
BAL = {k: w["initial"] for k, w in GENERAL.items()}
CARD_CYCLE = {k: 0.0 for k in CARDS}   # charges in the open billing cycle
CARD_STATEMENT = {k: float(w["initial_used"]) for k, w in CARDS.items()}


def rtime(lo=8, hi=21):
    return f"{rng.randint(lo, hi):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def add(d, ttype, amount, wallet, sys_key=None, dest=None, note=None, t=None):
    amount = round(float(amount), 2)
    eff_w = 1 if ttype in ("income", "creditCardCashback") else -1
    eff_d = 1 if dest else None
    cat = CAT[(wallet, sys_key)] if sys_key else None
    TXNS.append(dict(
        sync_id=sid(f"tx:{len(TXNS)}:{d}:{wallet}:{ttype}:{amount}"),
        type=ttype, amount=amount, date=f"{d.isoformat()}T{t or rtime()}+07:00",
        note=note, wallet=wallet, dest=dest, eff_w=eff_w, eff_d=eff_d, cat=cat))
    # Running balances so the fixture never shows a negative cash/e-wallet.
    if wallet in BAL:
        BAL[wallet] += eff_w * amount
    if dest in BAL:
        BAL[dest] += amount
    if wallet in CARD_CYCLE:
        CARD_CYCLE[wallet] += -eff_w * amount


def ensure(d, wallet, need):
    """Top up cash / e-wallet from the bank before a spend would overdraw it."""
    if wallet == "cash" and BAL["cash"] < need:
        add(d, "transfer", 2000, "kbank", dest="cash", note="กดเงิน ATM", t="08:05:00")
    if wallet == "truemoney" and BAL["truemoney"] < need:
        add(d, "transfer", 500, "kbank", dest="truemoney", note="เติม TrueMoney", t="07:50:00")


def spend(d, wallet, sys_key, amount, note, t=None):
    ensure(d, wallet, amount)
    add(d, "expense", amount, wallet, sys_key, note=note, t=t)


COFFEE = ["Café Amazon", "Starbucks", "อเมซอน ใต้ออฟฟิศ", "All Café 7-Eleven"]
LUNCH = ["ข้าวมันไก่", "ข้าวกะเพรา", "ก๋วยเตี๋ยว", "ข้าวแกง", "ส้มตำ ไก่ย่าง", "ราดหน้า"]
SOCIAL = ["เลี้ยงเพื่อนร่วมงาน", "หมูกระทะกับเพื่อน", "ปาร์ตี้วันเกิดเพื่อน", "นั่งบาร์หลังเลิกงาน"]
GROC = ["Lotus's", "Big C", "Tops", "Makro"]


def gen_transactions():
    # One-off events (month offset from START, day) that give analyst questions a
    # story: Chiang Mai trip, a gadget month, a bonus month, a spending spike in
    # the latest month. Applied inside the day loop so card cycles include them.
    events = {}

    def on(month_off, day, ttype, wallet, sys_key, amount, note):
        dd = add_months(START.replace(day=1), month_off).replace(day=day)
        events.setdefault(dd, []).append((ttype, amount, wallet, sys_key, note))

    on(1, 3, "expense", "ktc", "cc_shopping", 2490, "เสื้อผ้า Uniqlo")
    on(2, 9, "expense", "kbank", "health", 1200, "หาหมอ คลินิก + ยา")
    on(2, 30, "income", "kbank", "income_bonus", 30000, "โบนัสกลางปี")
    on(3, 12, "expense", "ktc", "cc_shopping", 8990, "AirPods Pro")
    on(4, 14, "expense", "ktc", "cc_travel", 3200, "ตั๋วเครื่องบิน เชียงใหม่")
    on(4, 15, "expense", "ktc", "cc_travel", 4800, "โรงแรม เชียงใหม่ 2 คืน")
    on(4, 16, "expense", "ktc", "cc_food", 1350, "ข้าวซอย + คาเฟ่ เชียงใหม่")
    on(4, 18, "income", "kbank", "income_overtime", 6500, "ค่า OT โปรเจกต์ด่วน")
    on(5, 6, "expense", "ktc", "cc_shopping", 16900, "iPad Air")
    on(5, 13, "expense", "kbank", "shopping_clothing", 3290, "รองเท้าวิ่ง Nike")
    on(5, 20, "expense", "kbank", "entertainment_social", 2400, "งานแต่งเพื่อน (ซองช่วยงาน)")

    d = START
    while d <= TODAY:
        for ttype, amount, wallet, sys_key, note in events.get(d, []):
            add(d, ttype, amount, wallet, sys_key, note=note)
        wd = d.weekday()  # 0 = Mon
        m_idx = (d.year - START.year) * 12 + d.month - START.month
        hot = d.month in (4, 5)

        # ── monthly fixed items ──
        if d.day == 1:
            add(d, "expense", 9500, "kbank", "home", note="ค่าเช่าคอนโด", t="09:00:00")
            add(d, "transfer", 1000, "kbank", dest="truemoney", note="เติม TrueMoney", t="09:10:00")
        if d.day == 3:
            add(d, "expense", 1290, "kbank", "health_fitness", note="ค่าสมาชิก Fitness First", t="10:00:00")
        if d.day == 5:
            amt = rng.randint(1500, 1850) if hot else rng.randint(950, 1300)
            add(d, "expense", amt, "kbank", "bills_utilities", note="ค่าไฟ MEA", t="12:30:00")
        if d.day == 8:
            add(d, "expense", 899, "kbank", "bills", note="ค่าเน็ตบ้าน + มือถือ AIS", t="12:00:00")
        if d.day == 10:
            add(d, "expense", 419, "scb", "cc_subscription", note="Netflix", t="03:00:00")
        if d.day == 12:
            add(d, "expense", 149, "scb", "cc_subscription", note="Spotify", t="03:00:00")
        if d.day == 15:
            add(d, "expense", 35, "scb", "cc_subscription", note="iCloud 50GB", t="03:00:00")
            add(d, "transfer", 1000, "kbank", dest="truemoney", note="เติม TrueMoney", t="09:10:00")
            spend(d, "cash", "home_cleaning", rng.choice([289, 345, 412]), "น้ำยาซักผ้า ของใช้ในบ้าน")
        if d.day == 22:
            add(d, "creditCardCashback", 150, "ktc", "cc_cashback", note="KTC cashback", t="06:00:00")
        if d.day == 25:
            add(d, "income", 42000, "kbank", "income_salary", note="เงินเดือน", t="06:30:00")
        if d.day == 26:
            add(d, "goalDeposit", 5000, "kbank", dest="japan", note="เก็บเงินเที่ยวญี่ปุ่น", t="08:00:00")
        if d.day == 27:
            add(d, "goalDeposit", 3000, "kbank", dest="emergency", note="เงินสำรองฉุกเฉิน", t="08:00:00")

        # ── credit card statements: close on bill day, pay in full on due day ──
        for k, c in CARDS.items():
            if d.day == c["bill"]:
                CARD_STATEMENT[k] += CARD_CYCLE[k]
                CARD_CYCLE[k] = 0.0
            if d.day == c["due"] and CARD_STATEMENT[k] > 0:
                add(d, "creditCardPay", round(CARD_STATEMENT[k], 2), "kbank", dest=k,
                    note=f"จ่ายบัตร {c['name']}", t="09:30:00")
                CARD_STATEMENT[k] = 0.0

        # ── weekday routine ──
        if wd < 5:
            if rng.random() < 0.55:
                spend(d, "truemoney", "food_coffee", rng.choice([55, 60, 65, 75, 95, 120]),
                      rng.choice(COFFEE), t=f"08:{rng.randint(10, 50)}:00")
            if rng.random() < 0.6:
                spend(d, "cash", "food_restaurant", rng.choice([50, 60, 65, 70, 80, 120, 150]),
                      rng.choice(LUNCH), t=f"12:{rng.randint(5, 45)}:00")
            if rng.random() < 0.4:
                spend(d, "truemoney", "transport_bts", rng.choice([32, 44, 52, 62]), "BTS",
                      t=f"18:{rng.randint(10, 50)}:00")
            if rng.random() < 0.08:
                spend(d, "truemoney", "transport_taxi", rng.randint(90, 240), "Grab กลับบ้าน (ฝนตก)",
                      t=f"19:{rng.randint(0, 59):02d}:00")
            if wd == 4 and rng.random() < (0.55 if m_idx == 5 else 0.35):
                add(d, "expense", rng.randint(550, 1700), "kbank", "entertainment_social",
                    note=rng.choice(SOCIAL), t="20:30:00")
        # ── weekend ──
        else:
            if wd == 5 and rng.random() < 0.75:
                add(d, "expense", rng.randint(420, 1450), "ktc", "cc_groceries",
                    note=rng.choice(GROC), t="11:00:00")
            if wd == 6 and rng.random() < 0.5:
                add(d, "expense", rng.randint(320, 980), "ktc", "cc_food",
                    note=rng.choice(["ชาบูกับแฟน", "MK สุกี้", "Sizzler", "อาหารญี่ปุ่น"]), t="18:30:00")
            if wd == 5 and d.day in range(8, 22) and rng.random() < 0.6:
                add(d, "expense", 280, "ktc", "cc_entertainment", note="ดูหนัง Major", t="15:00:00")
        d += timedelta(days=1)



def txn_sql():
    rows = []
    for t in TXNS:
        c = t["cat"]
        rows.append(
            "(" + ", ".join([
                esc(t["sync_id"]), esc(USER_ID), esc(t["type"]), str(t["amount"]),
                esc(t["date"]), esc(t["note"]), "'THB'", "'฿'",
                esc(c["sync_id"]) if c else "NULL", esc(sid("wallet:" + t["wallet"])),
                esc(sid("wallet:" + t["dest"])) if t["dest"] else "NULL",
                str(t["eff_w"]), str(t["eff_d"]) if t["eff_d"] else "NULL",
                (esc(c["icon"]) + "::jsonb") if c else "NULL", esc(c["name"]) if c else "NULL",
                "'confirmed'", "true", "1", "false", "false", esc(t["date"]), "NOW()",
            ]) + ")")
    cols = ("(sync_id, created_by_user_id, type, amount, date, note, currency_code, "
            "currency_symbol, category_sync_id, wallet_sync_id, destination_wallet_sync_id, "
            "effect_on_wallet, effect_on_destination, icon, category_name, status, "
            "include_in_report, version, is_deleted, is_recurring, created_at, updated_at)")
    out = []
    for i in range(0, len(rows), 200):
        out.append(f"INSERT INTO transactions {cols} VALUES\n" + ",\n".join(rows[i:i + 200]) + ";")
    return out


def budget_sql():
    """One repeating monthly food budget across all spend wallets (current month)."""
    bsid = sid("budget:food")
    ms = TODAY.replace(day=1)
    me = TODAY.replace(day=calendar.monthrange(TODAY.year, TODAY.month)[1])
    rows = [
        "INSERT INTO budgets (sync_id, user_id, owner_id, name, amount, period, is_repeat, goal, icon, "
        "start_date, end_date, sync_status, currency, currency_symbol, mode) VALUES ("
        f"{esc(bsid)}, {esc(USER_ID)}, {esc(USER_ID)}, 'งบอาหาร', 7000, 'monthly', true, "
        f"'คุมค่ากินไม่ให้เกิน 7,000 บาท/เดือน', "
        f"{esc(icon_json('assets/cetegory_icons/food.png', '0xFFF9E6AE'))}, "
        f"{esc(ms.isoformat() + ' 00:00:00')}, {esc(me.isoformat() + ' 23:59:59')}, 'synced', "
        "'THB', '฿', 'expenseOnly');"]
    wallets = [("kbank", "general"), ("cash", "general"), ("truemoney", "general"),
               ("ktc", "creditcard"), ("scb", "creditcard")]
    for wk, kind in wallets:
        rows.append("INSERT INTO budget_wallets (budget_sync_id, wallet_sync_id, wallet_type) VALUES "
                    f"({esc(bsid)}, {esc(sid('wallet:' + wk))}, {esc(kind)});")
    keys = {"general": ["food", "food_restaurant", "food_coffee", "food_groceries"],
            "creditcard": ["cc_food", "cc_groceries"]}
    for wk, kind in wallets:
        for k in keys[kind]:
            rows.append("INSERT INTO budget_categories (budget_sync_id, category_sync_id, wallet_sync_id) "
                        f"VALUES ({esc(bsid)}, {esc(CAT[(wk, k)]['sync_id'])}, {esc(sid('wallet:' + wk))});")
    return rows


WIPE_SQL = """
-- Wipe finance data for ALL users (children first).
DELETE FROM transaction_tags;
DELETE FROM tags;
DELETE FROM transaction_audit_logs;
DELETE FROM transactions;
DELETE FROM recurring_transaction_logs;
DELETE FROM recurring_transactions;
DELETE FROM recurring_items;
DELETE FROM budget_categories;
DELETE FROM budget_wallets;
DELETE FROM budget_members;
DELETE FROM budgets;
DELETE FROM obligation_transactions;
DELETE FROM obligation_plan_versions;
DELETE FROM obligation_wallets;
UPDATE categories SET parent_id = NULL;
DELETE FROM categories;
DELETE FROM wallet_share_codes;
DELETE FROM wallet_members;
DELETE FROM goal_wallets;
DELETE FROM creditcard_wallets;
DELETE FROM general_wallets;
"""


def build_sql():
    seed = load_seed_categories()
    wallets = build_wallets()
    cats = build_categories(seed)
    gen_transactions()
    for k, v in BAL.items():
        if v < 0:
            raise SystemExit(f"fixture bug: {k} goes negative ({v})")
    return "\n".join(["BEGIN;", WIPE_SQL, *wallets, *cats, *budget_sql(), *txn_sql(), "COMMIT;"])


def db_url():
    for line in open(os.path.join(ROOT, "backend/.env.dev"), encoding="utf-8"):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("DATABASE_URL not found in backend/.env.dev")


def psql(url, *args, stdin=None):
    return subprocess.run(["psql", url, "-v", "ON_ERROR_STOP=1", *args], input=stdin,
                          capture_output=True, text=True, check=True).stdout


def main():
    global USER_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--email", default="ittipong.it@gmail.com")
    a = ap.parse_args()
    url = db_url()
    USER_ID = psql(url, "-tAc", f"SELECT id FROM users WHERE email = {esc(a.email)}").strip()
    if not USER_ID:
        raise SystemExit(f"no user with email {a.email}")
    sql = build_sql()
    if not a.apply:
        print(sql)
        return
    psql(url, "-q", stdin=sql)
    print(f"seeded user {USER_ID}: {len(TXNS)} transactions, {len(CAT)} categories, "
          f"window {START}..{TODAY}", file=sys.stderr)


if __name__ == "__main__":
    main()
