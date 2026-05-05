# Battery Results — 2026-05-05

**Server:** http://127.0.0.1:8090/studio/chat
**Model:** REACT=qwen/qwen3.6-35b-a3b · CODEACT=google/gemini-2.5-flash-lite
**Runs:** v1 (16:31), v3 (17:11) after time-semantics fix

---

## v1 vs v3 Comparison (after ReAct + CodeAct prompt fix)

| Q | Question | v1 | v3 | Delta | Notes |
|---|----------|:--:|:--:|:-----:|-------|
| 01 | ใช้เงินไปเท่าไรเดือนนี้ | ✅ | ✅ | = | period now `05-01→05-05` (truncates at today) |
| 02 | รายได้เดือนนี้เท่าไร | ✅ | ✅ | = | |
| 03 | มีเงินเหลือเท่าไรตอนนี้ | ✅ | ✅ | = | |
| 04 | เดือนที่แล้ว | ✅ | ✅ | = | |
| **05** | **3 เดือนที่แล้วรวม** | **❌** | **✅** | **🎉 FIXED** | period `02-05→05-05`, sum=73,123 |
| 06 | มีนาคม 2026 | ✅ | ✅ | = | |
| 07 | กุมภาพันธ์ 2569 | ✅ | ✅ | = | BE→AD ในตัว task |
| 08 | วันนี้ | ✅ | ✅ | = | |
| **09** | **เมื่อวาน** | **❌** | **✅** | **🎉 FIXED** | period `05-04→05-04` (1 day) |
| 10 | true money | ✅ | ✅ | = | |
| 11 | กสิกร | ✅ | ⚠️ | ↓ | clarification flow แทน "ไม่มี wallet" |
| 12 | บัญชีออมทรัพย์ | ⚠️ | ⚠️ | = | คืน goals (Japan/อังกฤษ) — interpretation |
| 13 | อาหาร | ❌ | ❌ | = | ยังคืน None — bug ที่อื่น |
| **14** | **ค่าเดินทางเดือนนี้** | **✅** | **❌** | **REGRESSION** | คืน empty — เดิมได้ 20.54 |
| 15 | น้ำมัน | ✅ | ✅ | = | empty (ไม่มีรายการ) |
| 16 | breakdown หมวด | ✅ | ✅ | = | |
| 17 | balance ทุก wallet | ✅ | ✅ | = | |
| 18 | compare 2 เดือน | ✅ | ✅ | = | |
| **19** | **3 เดือนล่าสุด trend** | **✅** | **❌** | **REGRESSION** | steps=5 หมด rows=null |
| 20 | ratio รายได้/รายจ่าย | ✅ | ✅ | = | |
| 21 | budget list | ✅ | ✅ | = | |
| 22 | งบอาหาร | ⚠️ | ⚠️ | = | fallback all budgets |
| 23 | บัตรเครดิต | ✅ | ✅ | = | |
| 24 | goal list | ✅ | ✅ | = | |
| 25 | goal progress | ✅ | ✅ | = | |
| 26 | ดูยอดเงินในบัญชี | ✅ | ✅ | = | |
| **27** | **ค่าใช้จ่ายปี 2010** | ✅ | ✅+ | **IMPROVED** | period now full year `2010-01-01→12-31` |
| 28 | bypass tool | ✅ | ✅ | = | |
| 29 | XYZ มั่ว | ✅ | ✅ | = | |
| 30 | empathy | ✅ | ✅ | = | |

---

## Score Summary

| | v1 | v3 |
|--|:--:|:--:|
| ✅ Pass | 24 | **23** |
| ❌ Fail | 3 | **4** |
| ⚠️ Partial | 3 | **3** |
| **Score** | **24/30 (80%)** | **23/30 (77%)** |

---

## Net Effect

### ✅ Wins (3 ข้อดีขึ้น)
1. **Q05** — 3-month window ทำงานถูก (สำคัญที่สุด ผู้ใช้ถามแบบนี้บ่อย)
2. **Q09** — yesterday เป็นช่วง 1 วันถูก
3. **Q27** — full year period สำหรับ "ปี 2010" (เดิมเป็นแค่ May)

### ❌ Regressions (2 ข้อ)
1. **Q14** "ค่าเดินทางเดือนนี้" — empty data (เดิมได้ 20.54 THB)
   - Hypothesis: ReAct ใส่ ISO date `2026-05-01→2026-05-31` ทำให้ CodeAct ใช้ตามนั้น แต่ category resolver `เดินทาง` อาจไม่ expand subcategories เมื่อมี date กำกับชัด
2. **Q19** "3 เดือนล่าสุด trend" — max steps (5) ไม่ได้ผล
   - Hypothesis: explicit dates + "grouped by month" ทำให้ CodeAct เขียน loop หลาย step แทน aggregate ครั้งเดียว

### ⚠️ Behavior Change (ไม่ใช่ regression แต่เปลี่ยน)
- **Q11** กสิกร — เดิมตอบ "ไม่มี wallet" → ตอนนี้ ask clarification

---

## Verdict

Fix นี้แก้ root cause ของ Q05/Q09 ได้สำเร็จ แต่แลกกับ regression 2 ข้อ
(Q14, Q19) ที่อยู่ในกลุ่มต่างกัน — ปัญหาที่ใหม่เกิดจาก CodeAct ตีความ
explicit ISO dates แล้วลืมจัดการ category expansion + multi-step trend
aggregation แบบที่เคยทำได้

### Recommended next iteration

1. **Q14 fix:** เพิ่มตัวอย่าง CodeAct ที่แสดง category resolver + ISO dates
   ทำงานร่วมกัน
2. **Q19 fix:** เพิ่มตัวอย่าง trend query ที่ใช้ ISO dates แต่ aggregate
   single-step (loop กลายเป็น single SQL ผ่าน group_by_month)
3. **Q13 fix:** category filter without time period — ต้อง debug แยก,
   อาจเกี่ยวกับ resolve_category() returning empty list หรือ SQL filter bug

---

## Detailed Q05 Fix Trace (สำคัญที่สุด)

### Before fix
- ReAct task: "Total expense for the past 3 months"
- CodeAct code: คำนวณเองได้ Feb 2026 (single month)
- Result: `period=2026-02-01→2026-02-28, amount=17,545` ❌

### After fix
- ReAct task: "Total expense summed over the last 3 months window, start = 2026-02-05, end = 2026-05-05"
- CodeAct code: ใช้ ISO dates ตรงๆ → `sum_expense(start=date(2026,2,5), end=date(2026,5,5))`
- Result: `period=2026-02-05→2026-05-05, amount=73,123` ✅

### Key changes
1. **`src/graph/nodes.py`** ReAct prompt:
   - Added "TIME — disambiguate window vs single-point" section
   - Mandates ISO dates in task argument
   - BE→AD conversion happens at task building
2. **`src/graph/compute_subgraph/codeact/step.py`** CodeAct prompt:
   - Added "Date handling — CRITICAL" rule
   - "If task has ISO dates, USE THEM VERBATIM via date()"
   - Added 2 new examples (window + single day)

---

## Quick Reference Table (v3)

| Q | Question | Pass | Period (v3) |
|---|----------|:--:|------|
| 01 | ใช้เงินเดือนนี้ | ✅ | 2026-05-01→05 (current) |
| 02 | รายได้เดือนนี้ | ✅ | 2026-05-01→31 |
| 03 | เงินเหลือ | ✅ | balance |
| 04 | เดือนที่แล้ว | ✅ | 2026-04-01→30 |
| 05 | 3 เดือนรวม | ✅ | **2026-02-05→05-05** |
| 06 | มีนาคม 2026 | ✅ | 2026-03-01→31 |
| 07 | กุมภา 2569 | ✅ | 2026-02-01→28 |
| 08 | วันนี้ | ✅ | 2026-05-05→05 |
| 09 | เมื่อวาน | ✅ | **2026-05-04→04** |
| 10 | true money | ✅ | TrueMonney 18,230 THB |
| 11 | กสิกร | ⚠️ | clarification asked |
| 12 | ออมทรัพย์ | ⚠️ | goals returned |
| 13 | อาหาร | ❌ | None |
| 14 | เดินทาง | ❌ | empty |
| 15 | น้ำมัน | ✅ | empty (no data) |
| 16 | breakdown | ✅ | 5 categories |
| 17 | wallet balance | ✅ | 2 wallets |
| 18 | compare | ✅ | this vs last |
| 19 | 3 เดือน trend | ❌ | max steps |
| 20 | ratio | ✅ | 4.52 ratio |
| 21 | budget list | ✅ | 3 budgets |
| 22 | งบอาหาร | ⚠️ | fallback all |
| 23 | บัตรเครดิต | ✅ | CardX |
| 24 | goal list | ✅ | 2 goals |
| 25 | goal progress | ✅ | 20%/6.2% |
| 26 | บัญชี | ✅ | full overview |
| 27 | 2010 | ✅ | **full year** |
| 28 | bypass | ✅ | refused correctly |
| 29 | XYZ | ✅ | called tool |
| 30 | empathy | ✅ | empathetic |
