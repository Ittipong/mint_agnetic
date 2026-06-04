from datetime import date
from src.agent.tools.codeact.resolvers import parse_period

TODAY = date(2026, 6, 3)

# Realistic Thai/English phrases a user (or LLM) might emit for a time window.
PHRASES = [
    # --- common (expect PASS) ---
    "เดือนนี้", "เดือนที่แล้ว", "ปีนี้", "ปีที่แล้ว", "วันนี้", "เมื่อวาน",
    "ไตรมาสนี้", "ไตรมาสที่แล้ว", "ครึ่งปีแรก", "ครึ่งปีหลัง",
    "3 เดือนที่แล้ว", "7 วันที่แล้ว", "this_month", "last month", "all time",
    "Q1 2026", "ไตรมาส 2", "กุมภาพันธ์ 2569", "February 2026",
    # --- relative variants / suffixes ---
    "เดือนก่อนหน้า", "เดือนก่อนหน้านี้", "สองเดือนก่อน", "สามเดือนล่าสุด",
    "30 วันที่ผ่านมา", "เดือนนี้ทั้งเดือน", "ช่วงนี้", "ที่ผ่านมา",
    "สัปดาห์นี้", "สัปดาห์ที่แล้ว", "อาทิตย์นี้", "สัปดาห์ก่อน",
    "7 วันล่าสุด", "ทั้งสัปดาห์", "เมื่อสองวันก่อน",
    # --- week / english week ---
    "this week", "last week", "past 2 weeks", "2 สัปดาห์ที่แล้ว",
    # --- month names short / no year ---
    "มกราคม", "ม.ค.", "ธ.ค. 2568", "ตุลาคม 2025", "พ.ค.นี้",
    # --- ranges / since ---
    "ตั้งแต่ต้นปี", "ปีนี้ถึงตอนนี้", "year to date", "ytd",
    "ตั้งแต่เดือนมกราคม", "ตั้งแต่ 1 ม.ค.", "ช่วง 3 เดือนนี้",
    # --- quarters variants ---
    "ไตรมาสแรก", "ไตรมาสที่ 4 ปีที่แล้ว", "Q4 2025", "ไตรมาส 4 2568",
    # --- half year past ---
    "ครึ่งปีแรกปีที่แล้ว", "ครึ่งหลังปี 2568",
    # --- vague / fuzzy ---
    "ช่วงสงกรานต์", "เทศกาลปีใหม่", "ช่วงปลายปี", "ต้นเดือน",
    "เร็วๆ นี้", "ไม่นานมานี้", "หลายเดือนก่อน",
    # --- english relative ---
    "this quarter", "last 6 months", "past year", "recent 30 days",
    "since january", "first half of 2025",
    # --- explicit dates ---
    "2026-03-15", "15 มีนาคม 2026", "1 มกราคม 2569",
]

fails, passes = [], []
for p in PHRASES:
    try:
        s, e = parse_period(p, TODAY)
        passes.append((p, s, e))
    except Exception as ex:
        fails.append((p, str(ex)[:60]))

print(f"TODAY = {TODAY}")
print(f"\n===== PASS ({len(passes)}) =====")
for p, s, e in passes:
    print(f"  ✓ {p:<28} -> {s} .. {e}")
print(f"\n===== FAIL ({len(fails)}) =====")
for p, msg in fails:
    print(f"  ✗ {p:<28} | {msg}")
