"""NEW USER PLAYBOOK checks (fresh user: 1 wallet 'กระเป๋าหลัก' 5,000 THB, 0 txns).
N1 no hollow numbers · N2 no dead ends · chips are doable · starter plan maths."""
import glob, json, re, sys

HOLLOW = [r"เฉลี่ย[^\n]{0,20}\b0 บาท", r"\*\*0 บาท\*\*", r"11,667", r"เงินสำรอง[^\n]{0,40}\*\*0 บาท",
          r"3 เดือนที่ผ่านมา"]
NEXT_STEP = [r"รายได้(ต่อเดือน|เดือนละ|ประมาณ)", r"กาแฟ 60", r"พิมพ์", r"วางแผนเริ่มต้น", r"สลิป",
             r"ค่าห้อง", r"รายจ่ายประจำ", r"จำ", r"ยืนยัน"]
APP_ONLY = re.compile(r"^\s*(เพิ่ม|สร้าง|ตั้ง|ลบ|แก้ไข|แก้|ย้าย|โอน)\s*(กระเป๋า|บัญชี|บัตร|เป้า|งบ|รายการประจำ|หมวด|แท็ก|รายการ)")
NO_DATA_Q = {"N03_chip_top_spend", "N04_chip_health", "N05_chip_emergency", "N06_chip_salary_split",
             "N10_cant_save", "N11_where_money", "N12_budget"}
fails = checks = 0
for f in sorted(glob.glob(sys.argv[1] + "/N*.json")):
    r = json.load(open(f))
    for i, t in enumerate(r["turns"]):
        if t["msg"].startswith("__"):
            continue
        checks += 1
        a = t["answer"]; errs = []
        if t["errors"]:
            errs.append("error " + t["errors"][0][:60])
        for pat in HOLLOW:
            if re.search(pat, a):
                errs.append(f"hollow number /{pat}/")
        if r["id"] in NO_DATA_Q and not any(re.search(p, a) for p in NEXT_STEP):
            errs.append("dead end (no next step)")
        for b in t["blocks"]:
            if b.get("type") == "suggestions":
                for c in b.get("items") or []:
                    if APP_ONLY.match(c.get("label", "")) or APP_ONLY.match(c.get("send", "")):
                        errs.append(f"app-only chip {c.get('label')!r}")
        if r["id"] == "N02_capability" or r["id"] == "N14_capability_short":
            if "ข้อจำกัด" in a:
                errs.append("limitations section in first capability answer")
        if r["id"] == "N00_wallet_created" and "รายได้ต่อเดือน" not in a:
            errs.append("welcome missing the starter question")
        if r["id"] == "N13_starter_flow" and i == 2:
            for n in ("1,000", "5,000", "60,000"):
                if n not in a:
                    errs.append(f"starter plan missing {n}")
        if r["id"] == "N08_first_add" and i == 0 and "รายการแรก" not in a:
            errs.append("first record not celebrated")
        if errs:
            fails += 1
            print(f"FAIL {r['id']}[{i}] {t['msg'][:30]!r}: " + "; ".join(errs))
print(f"\n{checks - fails}/{checks} new-user checks passed")
