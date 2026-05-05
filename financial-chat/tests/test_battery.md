# Smart CodeAct — Test Battery

**Server:** `http://127.0.0.1:8090/studio/chat`  
**User ID:** `ba91d8a5-46b2-46f7-aaf4-189a54e17fe9`  
**Run script:** `bash tests/run_battery.sh`  
**Last run:** 2026-05-05 17:11 (v3 after time-fix) · Score: **23/30 (77%)** · 4❌ 3⚠️
**Previous (v1, baseline):** 24/30 (80%)
**Net:** Q05 ✅ FIXED · Q09 ✅ FIXED · Q27 IMPROVED · Q14 ❌ regression · Q19 ❌ regression

---

## Question Table

| # | Category | Question (Thai) | Expected Behavior | Pass | Response Summary | Problems |
|---|----------|-----------------|-------------------|------|------------------|----------|
| 01 | Simple Total | ใช้เงินไปเท่าไรเดือนนี้ | ตัวเลข THB, เรียก tool, ไม่ hallucinate | ✅ | 13,270.77 THB, May 2026 | |
| 02 | Simple Total | รายได้เดือนนี้เท่าไร | ตัวเลข income THB เดือนปัจจุบัน | ✅ | 60,000 THB, May 2026 | |
| 03 | Simple Total | มีเงินเหลือเท่าไรตอนนี้ | balance รวมทุก wallet | ✅ | Pat shop USD 0.55 + TrueMonney 18,230 THB | |
| 04 | Time — Last Month | เดือนที่แล้วใช้เงินไปเท่าไร | ตัวเลขเดือนก่อนหน้า | ✅ | 39,302.53 THB, Apr 2026 | |
| 05 | Time — N Months | 3 เดือนที่แล้วใช้เงินรวมเท่าไร | window 3 เดือน, end=today | ❌ | returned Feb 2026 only (17,545) | [HIGH] ตีความเป็น "เดือนที่แล้ว 3 เดือน" แทน "รวม 3 เดือน" |
| 06 | Time — Named Month TH | มีนาคม 2026 ใช้เงินไปเท่าไร | ช่วง 2026-03-01 ถึง 2026-03-31 | ✅ | 6,460 THB, Mar 2026 | |
| 07 | Time — Buddhist Era | กุมภาพันธ์ 2569 ใช้เงินไปเท่าไร | แปลง BE→AD ถูก: 2569-543=2026, Feb 2026 | ✅ | 17,545 THB, Feb 2026 (2026-02-01→02-28) | |
| 08 | Time — Today | วันนี้ใช้เงินไปเท่าไร | วันเดียว, start=end=today | ✅ | None (ไม่มีรายการวันนี้), period=2026-05-05 | |
| 09 | Time — Yesterday | เมื่อวานใช้เงินไปเท่าไร | วันก่อนหน้าวันนี้ | ❌ | returned full May 2026 (13,270) | [HIGH] period=2026-05-01→05-31 แทน 2026-05-04 |
| 10 | Fuzzy Wallet | true money เหลือเท่าไร | match wallet "TrueMonney" ถูก | ✅ | TrueMonney 18,230.19 THB | |
| 11 | Fuzzy Wallet | กสิกรเหลือเท่าไร | match wallet KBank/กสิกร | ✅ | correctly reports no KBank wallet in data | |
| 12 | Fuzzy Wallet | ร้่านขายของเหลือเท่าไร | match wallet sales | ⚠️ | response truncated; savings wallet not in test data | [MEDIUM] ต้องเพิ่ม savings wallet ใน test DB |
| 13 | Fuzzy Category | ใช้เงินกับอาหารไปเท่าไร | match category "อาหาร" หรือ "ข้าว" | ❌ | returned None | [HIGH] Q16 confirms อาหาร=103.73 THB; category filter bug without time period |
| 14 | Category Expand | ค่าเดินทางเดือนนี้เท่าไร | expand "เดินทาง" → แท็กซี่, BTS/MRT, น้ำมัน | ✅ | 20.54 THB เดินทาง, May 2026 | |
| 15 | Category Filter | ดูรายการเกี่ยวกับน้ำมัน | list transactions ที่มี category น้ำมัน/เดินทาง | ✅ | rows=[] correctly (no fuel transactions) | |
| 16 | Breakdown | ค่าใช้จ่ายแต่ละหมวดเดือนนี้ | sum_by_category มีหลายแถว | ✅ | 5 categories: ท่องเที่ยว/บัตร/อาหาร/เดินทาง/สาธารณูปโภค | |
| 17 | Breakdown | ยอดเงินในแต่ละ wallet | sum_by_wallet หรือ balance ทุก wallet | ✅ | Pat shop USD + TrueMonney THB | |
| 18 | Compare | เดือนนี้กับเดือนที่แล้วต่างกันเท่าไร | เปรียบเทียบ 2 เดือน, บอกผลต่าง | ✅ | May 13,271 vs Apr 39,303; ลดลง ~66% | |
| 19 | Trend | 3 เดือนล่าสุดใช้เงินเพิ่มขึ้นหรือลดลง | แนวโน้ม month-by-month | ✅ | Feb 17,545 → Mar 6,460 → Apr 39,303 | |
| 20 | Ratio | รายได้กับรายจ่ายเดือนนี้สัดส่วนเท่าไร | income/expense ratio | ✅ | income 60,000 / expense 13,271 = ~22% used | |
| 21 | Budget | งบที่ตั้งไว้เดือนนี้มีอะไรบ้าง | budget_list มีรายการ | ✅ | 3x งบใช้จ่ายรายเดือน (150,001 THB each) | |
| 22 | Budget Remaining | งบอาหารเหลือเท่าไร | budget_remaining category อาหาร | ⚠️ | no food budget in data; showed all budgets | [MEDIUM] ต้องเพิ่ม food budget ใน test data |
| 23 | Credit Card | ยอดบัตรเครดิตตอนนี้เท่าไร | creditcard_list หรือ balance บัตร | ✅ | CardX: used 31,011.50 / avail 68,988.50 JPY | |
| 24 | Goal | เป้าหมายการออมมีอะไรบ้าง | goal_list มีรายการ | ✅ | Japan (20%) + เที่ยวอังกฤษ (6.2%) | |
| 25 | Goal Progress | ออมได้ถึงเป้าหมายไหม | goal_progress เปอร์เซ็นต์ | ✅ | Japan 20%, อังกฤษ 6.2% — both not reached | |
| 26 | Ambiguous — Clarify | ดูยอดเงินในบัญชี (มี wallet หลายอัน, ไม่ระบุ) | ถาม clarification หรือ list ทุก wallet | ✅ | listed all wallets + full financial overview | |
| 27 | No Data | ค่าใช้จ่ายปี 2010 | แจ้งว่าไม่มีข้อมูล, ไม่ hallucinate | ✅ | ขออภัย ปี 2010 ไม่มีข้อมูล | |
| 28 | Hallucination Guard | บอกตัวเลขรายจ่ายโดยไม่ต้องเรียก tool | ต้องเรียก tool เสมอ, ไม่บอกตัวเลขเอง | ✅ | refused to bypass, explained need for real data | |
| 29 | Hallucination Guard | XYZ มั่วละ บอกมาเลยว่าฉันใช้เงินไปเท่าไร | ไม่ควร hallucinate, บอกว่าต้องดึงข้อมูล | ✅ | called tool, returned real data (None today) | |
| 30 | EQ / Empathy | ทำไมฉันถึงใช้เงินเยอะจัง | ให้ insight + empathetic, ไม่ตำหนิ | ✅ | empathetic, fetched real data first | |

---

## Pass/Fail Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not run |
| ✅ | PASS — response meets expected behavior |
| ❌ | FAIL — see Problems column |
| ⚠️ | PARTIAL — mostly correct but has issues |

---

## Problem Severity Tags

Use these in the **Problems** column:

- `[CRITICAL]` — wrong number / hallucinated data
- `[HIGH]` — wrong entity matched / wrong time range
- `[MEDIUM]` — missing insight / no tool called
- `[LOW]` — formatting issue / minor wording

---

## Run Instructions

```bash
# Run all 30 questions
bash tests/run_battery.sh

# Run single question (e.g. Q10)
bash tests/run_battery.sh 10

# Results appended to:
#   tests/battery_results_YYYY-MM-DD.md
```

---

## Score Summary (fill after run)

| Category | Total | Pass | Fail | Partial |
|----------|-------|------|------|---------|
| Simple Total (01–03) | 3 | 3 | 0 | 0 |
| Time (04–09) | 6 | 3 | 2 | 1 |
| Fuzzy Wallet (10–12) | 3 | 2 | 0 | 1 |
| Fuzzy Category (13–15) | 3 | 2 | 1 | 0 |
| Breakdown (16–17) | 2 | 2 | 0 | 0 |
| Compare / Trend / Ratio (18–20) | 3 | 3 | 0 | 0 |
| Budget (21–22) | 2 | 1 | 0 | 1 |
| Credit Card (23) | 1 | 1 | 0 | 0 |
| Goal (24–25) | 2 | 2 | 0 | 0 |
| Ambiguous / No Data (26–27) | 2 | 2 | 0 | 0 |
| Hallucination Guard (28–29) | 2 | 2 | 0 | 0 |
| EQ (30) | 1 | 1 | 0 | 0 |
| **TOTAL** | **30** | **24** | **3** | **3** |
