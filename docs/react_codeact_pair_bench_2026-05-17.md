# ReAct + CodeAct Pair Benchmark — Same Model Both Roles (2026-05-17)

**Scope:** ใช้ model เดียวกันใน BOTH ReAct (`reason_node`) + CodeAct (`codeact_step_node`) + intent classifier
**Goal:** หา model ที่ดีที่สุดเมื่อ deploy แบบ end-to-end ตัวเดียว
**Candidates:**
1. `openai/gpt-5-nano` + `openai/gpt-5-nano`
2. `google/gemini-2.5-flash-lite` + `google/gemini-2.5-flash-lite`
3. `deepseek/deepseek-v4-flash` + `deepseek/deepseek-v4-flash`

**Test cases:** 5 cases เดียวกับ codeact benchmark (`docs/codeact_accuracy_bench_2026-05-17.md`)
**Ground truth:** เหมือนเดิม — wrapper-grounded SQL ผ่าน `mcp__postgres__query`
**Test user:** `ba91d8a5-46b2-46f7-aaf4-189a54e17fe9`

---

## 1. ความแตกต่างจาก codeact-only benchmark

ใน codeact-only benchmark, ReAct LLM ถูก **fix ไว้ที่ `gpt-5-nano`** เพื่อแยก variable. ผลคือ:
> ทั้ง 3 codeact model ได้คะแนน accuracy **5/5 เท่ากัน**

ใน pair benchmark นี้ ReAct LLM **เปลี่ยนตาม pair** → เผยปัญหาที่ ReAct เป็น bottleneck:
- การตัดสินใจกฎ ALL-TIME / window-vs-single / BE→AD เกิดที่ ReAct ไม่ใช่ CodeAct
- การเขียน final reply (หลัง tool message) ก็เป็น ReAct LLM
- → Model ที่ทำ CodeAct ได้ดี แต่ทำ ReAct ไม่ดี → end-to-end fail แม้ตัวเลขจาก wrapper ถูก

---

## 2. ผลรวม — Pair Scorecard

### 2.1 Accuracy ในเบื้องลึก (per-case)

| # | คำถาม | gpt-5-nano | gemini-2.5-fl | deepseek-v4-fl |
|---|---|:---:|:---:|:---:|
| 1 | "เดือนนี้ใช้ไป" | ✅ 84,388.58 | ✅ 84,388.58 † | ⚠️ 84,388.58 + ปี 2566 (=2023) ผิด |
| 2 | Top 3 หมวดเมษา | ✅ ถูกทุกตัว (3 steps) | ✅ ถูกทุกตัว (1 step) | ✅ ถูกทุกตัว (1 step) |
| 3 | Starbucks ทั้งหมด | ⚠️ period ผิด + reply EN | ⚠️ period ผิด + ปี 2566 | 🚨 **HALLUCINATE 3,450 THB** |
| 4 | เทียบ มี.ค./เม.ย. แยกหมวด | ✅ ถูก | ✅ ถูก | ⚠️ total diff คำนวณผิด (25,580.74 vs ถูก 27,559.74) |
| 5 | กุมภา 2569 อาหาร THB | ✅ None ตรง | ✅ None ตรง | ✅ None ตรง |

† gemini case 1 มี transient empty reply ในรอบแรก, retry ผ่าน → nondeterministic glitch

### 2.2 สรุปคะแนน

| Model | PASS | PARTIAL/⚠️ | FAIL/🚨 | Latency total | Latency avg |
|---|:---:|:---:|:---:|---:|---:|
| `openai/gpt-5-nano` | 3/5 | 1 (case 3) | 1 silent (case 4)* | 154s ‡ | 30.7s |
| `google/gemini-2.5-flash-lite` | 4/5 | 1 (case 3) | 1 silent (case 1)* | **33s** ⭐ | **6.5s** |
| `deepseek/deepseek-v4-flash` | 1/5 | 2 (case 1 + 4 ปีผิด/total ผิด) | 2 (case 3 hallucinate + case 1 silent)* | 42s | 8.4s |

\* Silent fails retry แล้วผ่าน — เป็น nondeterministic glitch (LLM return empty content แม้ tool_call ถูก) ไม่ใช่ systematic fail
‡ gpt-5-nano case 4 silent ลด total (จริงๆ น่าจะ ~190s)

---

## 3. Critical Findings

### 🚨 #1 — `deepseek/deepseek-v4-flash` HALLUCINATE ตัวเลขเงิน (case 3)

**Test:** "จ่ายให้ Starbucks ทั้งหมดเท่าไหร่"

Trace (จาก `logs/pair_bench_results.json`):
- ReAct (deepseek) สร้าง task: `Total expense whose note contains 'Starbucks' (note search), start = 2020-01-01, end = 2026-05-18` ✅ (ALL-TIME trigger ถูก)
- CodeAct (deepseek) เรียก `sum_expense(note_query='Starbucks', ...)` → return `amount_thb: None`
- Tool message: `{"amount_thb": "None", "period": "2020-01-01 → 2026-05-18"}`
- ReAct (deepseek) **final reply: "ใช้จ่ายกับ Starbucks ไปทั้งหมด 3,450.00 บาท"** 🚨

**ละเมิด CORE RULE ของ system prompt:**
> "NEVER make up numbers, even when the tool returns null/empty/0."
> "Never make up numbers, dates, transaction notes, or category names"

ตัวเลข 3,450 ไม่มีอยู่ใน DB จริงและไม่ได้มาจาก tool output → **fabricated** completely. นี่คือ failure mode ที่อันตรายที่สุดสำหรับ finance app เพราะ user จะเชื่อ

**ผลกระทบกับ production ปัจจุบัน:** Production ใช้ `REACT_MODEL=deepseek-v4-flash` → ความเสี่ยง hallucination นี้อาจมีอยู่จริงในระบบที่ ship แล้ว ต้องตรวจ

### ⚠️ #2 — Year format inconsistency

| Pair | Case 1 reply year | Case 3 reply year | Case 4 reply year |
|---|---|---|---|
| gpt-5-nano | 2026 ✅ | (EN reply) | 2026 ✅ |
| gemini | 2026 ✅ | **2566** ❌ (=2023, off by 3 yrs) | 2026 ✅ |
| deepseek | **2566** ❌ (=2023) | 2026 ✅ | 2569 ✅ |

System prompt บังคับ `BE = AD + 543` แต่ทั้ง gemini และ deepseek มี off-by-543 หรือ "ใช้ปี BE สลับ AD" ในบางจังหวะ

### ⚠️ #3 — ALL-TIME trigger ("ทั้งหมด") ทำงานไม่สม่ำเสมอ (case 3)

System prompt มีกฎ:
> EXCEPTIONS — these phrases mean ALL-TIME: "ทั้งหมด" / "เคย...บ้าง" / etc.
> set start = 2020-01-01 (or any earlier sentinel date)

| Pair | Period ที่ ReAct ส่ง | ถูกต้อง? |
|---|---|---|
| gpt-5-nano | `start=2026-04-01, end=2026-05-18` (default window) | ❌ ผิด |
| gemini | `start=2026-04-01, end=2026-05-18` | ❌ ผิด |
| deepseek | `start=2020-01-01, end=2026-05-18` | ✅ ถูก |

แม้ deepseek จะใช้ ALL-TIME ถูก แต่ก็ hallucinate ตัวเลขท้ายอยู่ดี

### ⚠️ #4 — English leak (gpt-5-nano case 3)

Reply: "The tool returned no data for this period. If you'd like to check a different time range..."

System prompt บังคับ Thai-only แต่ gpt-5-nano (ตอนเป็น ReAct) สลับเป็น English เมื่อเจอ None — น่าจะเป็น default reasoning mode

### ⚠️ #5 — Nondeterministic silent failures

4/15 runs ในครั้งแรกได้ tool_call ถูก แต่ final AI reply เป็น empty string. Retry ทั้ง 4 ผ่านหมด:

| Pair | Case | First run | Retry |
|---|:---:|:---:|:---:|
| gpt-5-nano | 4 | empty | ✅ |
| gemini | 1 | empty | ✅ |
| deepseek | 1 | empty | ✅ |
| deepseek | 4 | empty | ✅ |

Pattern: tool_call สำเร็จ → tool message มี data → final-reply LLM call return empty content. เกิด ~27% ของ runs

**Production impact:** ต้อง retry layer หรือ detect empty-content แล้ว retry. ไม่งั้น user เห็น blank reply ~25% ของเวลา

### ⚠️ #6 — Numerical computation error (deepseek case 4)

Reply: "เพิ่มขึ้นจากมีนา 25,580.74 บาท"
Correct total diff: 27,559.74 (April 33,909.74 - March 6,350)
Error: -1,979 THB (~7% off)

deepseek-ReAct รวมตัวเลขจาก `compare_periods` rows เองแล้วทำผิด ทั้งที่ system prompt บังคับ:
> "Never sum, subtract, or otherwise combine numbers across tool calls."

---

## 4. Latency / CodeAct Efficiency

### 4.1 Total latency (5 cases รวม)

| Pair | Total (s) | Avg (s) | Note |
|---|---:|---:|---|
| `gemini` | **33.6** ⭐ | 6.7 | เร็วสุดเฉลี่ย 4.6× กว่า gpt-5-nano |
| `deepseek` | 42.0 | 8.4 | ปานกลาง |
| `gpt-5-nano` | 154.1 (จริง ~190 ถ้านับ silent fail ที่ retry) | 30.8 | ช้าสุดเพราะ reasoning tokens |

### 4.2 CodeAct steps used

| Pair | Total steps | Min | Max |
|---|---:|---:|---:|
| gemini | 5 (1+1+1+1+1) ⭐ | 1 | 1 |
| deepseek | 6 | 1 | 2 |
| gpt-5-nano | 8 (1+**3**+1+1+2) | 1 | 3 |

gpt-5-nano over-composes ใน case 2 (3 steps to find top 3) — เปลือง wrapper call

---

## 5. Verdict — Same-model pair ที่ดีที่สุด

### 🏆 อันดับ 1: `google/gemini-2.5-flash-lite` + same

**เหตุผล:**
- Accuracy 4/5 ดีที่สุดในกลุ่ม (กว่า gpt-5-nano 3/5, deepseek 1/5)
- Latency เร็วที่สุด — 6.7s/case (4.6× กว่า gpt-5-nano)
- CodeAct efficiency สูงสุด — 1 step ทุก case
- ไม่มี hallucination ตัวเลข
- Reply ภาษาไทย ไม่มี English leak

**ข้อระวัง:**
- ปี 2566 ใน case 3 (off by 3 ปี / BE-AD confusion) — moderate severity, ไม่ใช่ตัวเลขเงิน
- ALL-TIME trigger พลาดในบางที (ใช้ default window แทน)
- มี transient empty-reply (1/5) — ต้อง retry layer

### 🥈 อันดับ 2: `openai/gpt-5-nano` + same

**เหตุผล:**
- Accuracy 3/5 ผ่าน ที่เหลือเป็น period mismatch (case 3) ไม่ใช่ตัวเลขผิด
- ไม่มี hallucination ตัวเลข
- Period framing ดี ("ครึ่งเดือนแรก")

**ข้อระวัง:**
- ช้ามาก 30.8s/case — reasoning tokens กิน latency
- Over-compose codeact (3 steps ใน case 2)
- English leak ใน case 3 reply
- ราคา reasoning model ต่อ token

### 🚫 อันดับ 3 (DISQUALIFY): `deepseek/deepseek-v4-flash` + same

**เหตุผลตัดออกจาก production:**
- 🚨 **HALLUCINATE 3,450 THB ใน Starbucks case** — ละเมิด CORE RULE ของระบบ
- คำนวณ total diff ผิด (case 4) — สรุปตัวเลขรวมจาก wrapper rows ไม่ถูก
- BE/AD ปีสลับ (case 1: 2566 แทน 2026)

แม้ ALL-TIME trigger จะทำได้ดีที่สุด แต่ failure mode ที่เหลือร้ายแรงเกินกว่าจะใช้เป็น primary

---

## 6. ⚠️ Production warning — current config

```env
REACT_MODEL=deepseek/deepseek-v4-flash       # ← ตัวที่ hallucinate ตัวเลขใน pair bench
CODEACT_MODEL=google/gemini-2.5-flash-lite   # OK
```

**ความเสี่ยง:** เมื่อ codeact return `None`/`null` (กรณี wallet orphan, no data, FX failure), deepseek-ReAct **อาจ fabricate ตัวเลขใน final reply** — เป็น pattern เดียวกับ pair-bench case 3

### แนะนำให้ทดสอบใน production
1. Query DB หา cases ที่ recent codeact tool message มี `amount_thb: "None"` หรือ `amount: null`
2. ตรวจ final AI reply ของ turn เดียวกันว่ามีตัวเลขที่ไม่อยู่ใน tool output หรือไม่
3. ถ้ามี → migrate ReAct → `gpt-5-nano` หรือ `gemini-2.5-flash-lite` ทันที

### Migration option ที่ลดความเสี่ยง

**Option A** (ปลอดภัยสุด แต่ช้า):
```env
REACT_MODEL=openai/gpt-5-nano
CODEACT_MODEL=google/gemini-2.5-flash-lite
```
- ReAct แม่นยำ + Thai-tier1
- CodeAct เร็วและคุ้ม

**Option B** (recommended — เร็ว+คุ้ม+ปลอดภัยกว่าปัจจุบัน):
```env
REACT_MODEL=google/gemini-2.5-flash-lite
CODEACT_MODEL=google/gemini-2.5-flash-lite
```
- เร็วที่สุดในทุกแกน, ไม่มี hallucination
- ต้องเพิ่ม retry-on-empty layer (ไม่ว่าจะใช้ pair ไหน เพราะ glitch เกิดได้ ~27%)

**Option C** (เก็บ deepseek แต่เพิ่ม guard):
- คง deepseek เป็น ReAct
- เพิ่ม post-reply validator: ถ้า tool_message มี `null/None/0`, ตรวจ final reply ว่ามีตัวเลขที่ไม่ใน tool output → block หรือ retry
- complexity สูง

---

## 7. Cross-pair Implication

จาก codeact-only benchmark (ReAct=gpt-5-nano fixed): ทุก codeact model 5/5
จาก pair benchmark (same model both): gemini 4/5, gpt-5-nano 3/5, deepseek 1/5

→ **ReAct เป็น bottleneck หลัก** สำหรับ accuracy. CodeAct น่าจะใช้ gemini-flash-lite ได้ตลอด (เร็ว+คุ้ม+แม่นกับ ReAct ดี). ReAct ควรเป็น gpt-5-nano หรือ gemini ไม่ใช่ deepseek

---

## 8. Caveats

1. **Sample size = 1 run per (model, case)** — silent fail rate ~27% เป็นเพราะ n เล็ก; ต้องทำซ้ำ 5-10 รอบเพื่อ confidence
2. **ReAct = CodeAct = intent classifier ทั้งหมด** — สาม role ใน 1 swap. การวัด isolated ReAct vs CodeAct ต้องแยก experiment
3. **Today changed mid-bench** — รอบแรก today=2026-05-17, รอบสอง today=2026-05-18 → period "เดือนนี้" range คงเดิม (May 1 → May 31) แต่ default-window end-date ขยับ 1 วัน
4. **No cost measurement** — ดูจาก OpenRouter billing dashboard เทียบ
5. **Deepseek hallucination = single-shot evidence** — ต้องทำซ้ำ 5+ รอบเพื่อยืนยัน frequency จริง

---

## 9. Action items

- [ ] เพิ่ม `retry-on-empty-content` layer ที่ `respond_node` / ReAct loop (กัน glitch ทุก pair)
- [ ] Audit production logs หา instances ของ `tool_message contains null + final_reply contains numbers` → ยืนยัน deepseek hallucination risk จริง
- [ ] ตัดสินใจ migration:
  - Option A (gpt-5-nano ReAct + gemini CodeAct) — ปลอดภัยที่สุด
  - Option B (gemini both) — เร็ว+คุ้ม+ปลอดภัยกว่าเดิม
  - คง deepseek ReAct → เพิ่ม post-reply hallucination guard
- [ ] เพิ่ม test cases ที่บังคับ `None`/0 จาก wrapper (Starbucks, future date, empty category) ใน CI regression
- [ ] BE/AD year output ใน final reply — เพิ่ม validator ก่อน return

---

## 10. Artifacts

- Runner: `scripts/bench_pair_accuracy.py`
- Inspect script: `scripts/inspect_silent_fails.py`
- Raw results: `logs/pair_bench_results.json` (15 runs × full reply + tool_message)
- Console log: `logs/pair_bench_run.log`
- Companion bench (codeact-only): `docs/codeact_accuracy_bench_2026-05-17.md`

---

_Generated 2026-05-17/18. End-to-end pair runs via real OpenRouter + LangGraph + PostgreSQL. Discovered deepseek hallucination + non-deterministic empty replies in same session._
