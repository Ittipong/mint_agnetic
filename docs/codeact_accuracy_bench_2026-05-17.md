# CodeAct Accuracy Benchmark — Financial Q&A (2026-05-17)

**Target:** `mint_agentic/src/graph/compute_subgraph/codeact/step.py` (`codeact_step_node`)
**ReAct LLM (fixed):** `openai/gpt-5-nano`
**CodeAct candidates:** `openai/gpt-5-nano`, `google/gemini-2.5-flash-lite`, `deepseek/deepseek-v4-flash`
**Ground truth source:** Direct PostgreSQL queries via `mcp__postgres__query` ตรง schema/filter เดียวกับ `_sum_by_type` / `_compare_periods` ใน `sql_templates.py`
**Test user:** `ba91d8a5-46b2-46f7-aaf4-189a54e17fe9` (372 txs, 2017-12-30 → 2026-05-20)
**Date pinned:** 2026-05-17 (system clock match)

---

## 1. Methodology

### 1.1 End-to-end flow
แต่ละ test case รันผ่าน graph เต็ม:
```
HumanMessage → classify_intent → reason_node (gpt-5-nano)
   → tool_call(analyze_user_finances)
   → act_node → compute_subgraph → codeact_step_node (model under test)
   → wrapper SQL → result → ToolMessage → reason_node → final reply
```

โค้ดที่รัน: `scripts/bench_codeact_accuracy.py` (monkey-patch `src.llm.codeact_llm` + `src.llm.llm` ก่อนเรียก `build_graph().ainvoke()`)

### 1.2 Ground truth construction
SQL ที่ใช้สร้าง GT match กับ wrapper exactly:
- THB conversion ผ่าน `_AMOUNT_THB_EXPR` 4-tier (THB native → converted_amount ≠ amount → exchange_rate → currencies.rate → amount fallback)
- Filter `is_deleted=false AND status='confirmed' AND include_in_report=true`
- **Wallet user check** — JOIN `general_wallets`/`creditcard_wallets`/`goal_wallets` แล้วเช็ก `user_id` (ไม่ใช่ `transactions.created_by_user_id`)

> **ข้อค้นพบสำคัญ:** ผู้ใช้นี้มี orphan transactions (~10K THB ใน May 2026 และ ทั้งหมดของ Starbucks/Feb-อาหาร) ที่ wallet ownership ไม่ match user → wrapper return None. ทำให้ test case 3 และ 5 เป็น **honesty test** (model ต้องรายงาน 0/none ตรง ๆ ห้าม hallucinate)

### 1.3 Scoring rubric
| Score | Criteria |
|---|---|
| ✅ **PASS** | ตัวเลขตรงกับ wrapper output (±1% สำหรับ FX rounding) + Period ที่ระบุถูก + ไม่ hallucinate |
| ⚠️ **PARTIAL** | ตัวเลขถูก แต่ period mention ผิด / breakdown ขาด |
| ❌ **FAIL** | ตัวเลขผิด > 1% / เรียก wrapper ผิด / hallucinate ตัวเลข |
| 🚨 **ERROR** | Crash / MAX_STEPS / ClarificationNeeded ไม่จำเป็น |

---

## 2. Test Cases + Ground Truth

| # | คำถาม | Difficulty | สิ่งที่วัด |
|---|---|---|---|
| 1 | "เดือนนี้ใช้เงินไปเท่าไหร่" | ง่าย | This-month single-point, basic sum, expense type |
| 2 | "เดือนเมษายนใช้หมวดไหนเยอะสุด top 3" | กลาง | Breakdown + top-N + Thai month |
| 3 | "จ่ายให้ Starbucks ทั้งหมดเท่าไหร่" | กลาง-ยาก | ALL-TIME trigger ("ทั้งหมด") + note_query routing |
| 4 | "เปรียบเทียบรายจ่ายเดือนมีนากับเมษา แยกหมวด" | ยาก | `compare_periods` composition |
| 5 | "กุมภาพันธ์ 2569 ใช้เงินกับอาหารทั้งหมดคิดเป็นบาทเท่าไร" | ยากสุด | **BE→AD (2569→2026)** + category + `convert_to_thb` |

### Ground truth (wrapper-grounded)

**Case 1** — May 2026 total expense (THB)
- Wrapper expected: **84,388.58 THB / 35 confirmed+in-report txs ที่ wallet ยังเป็นของ user**

**Case 2** — Top 3 April 2026 categories
- 1) ซื้อวัตถุดิบ — 32,679.74 THB (1 tx, USD 1000 converted)
- 2) ค่าใช้จ่ายบัตร — 600 THB
- 3) ของใช้ส่วนตัว — 252 THB

**Case 3** — Starbucks ALL-TIME
- 4 txs ใน DB มี `note ILIKE '%Starbucks%'` แต่ทุก tx อยู่บน wallet orphan
- Wrapper expected: **None / 0** — model ต้องตอบว่าไม่พบ ห้ามแต่งตัวเลข

**Case 4** — March vs April by category
- March total: 6,350 (ร้านอาหาร 2,350, น้ำมัน 2,000, สุขภาพ 2,000)
- April total: 33,909.74 (ซื้อวัตถุดิบ 32,679.74, ค่าใช้จ่ายบัตร 600, ของใช้ส่วนตัว 252, ช้อปปิ้ง/อาหาร/อื่นๆ 126 each)
- Diff: +27,559.74 (largest = ซื้อวัตถุดิบ +32,089.74)

**Case 5** — Feb 2569 (=2026) อาหาร converted to THB
- "อาหาร" leaf only → wrapper: **None / 0** (wallet orphan)
- "อาหาร" expanded (parent+children กาแฟ/ซื้อวัตถุดิบ/ร้านอาหาร) → wrapper: **None / 0** เช่นกัน

---

## 3. Results — Per-Case Detail

### Case 1: "เดือนนี้ใช้เงินไปเท่าไหร่"

| Model | Tool output | Latency | CodeAct steps | Score |
|---|---|---:|---:|:---:|
| `gpt-5-nano` | 84,388.58 THB | 29.7s | 1 | ✅ PASS |
| `gemini-2.5-flash-lite` | 84,388.58 THB | **15.2s** | 1 | ✅ PASS |
| `deepseek-v4-flash` | 84,388.58 THB | 23.3s | 2 | ✅ PASS |

**สังเกต:** ทั้ง 3 model produce wrapper output ตรงกัน 100% — task argument ที่ ReAct ส่งเข้ามา `start=2026-05-01, end=2026-05-31` มี ISO dates ครบแล้ว, codeact แค่ส่งเข้า `sum_expense(convert_to_thb=True)` ตรงๆ. deepseek ใช้ 2 steps (over-iterating)

### Case 2: "เดือนเมษายนใช้หมวดไหนเยอะสุด top 3"

| Model | Tool output | Latency | CodeAct steps | Score |
|---|---|---:|---:|:---:|
| `gpt-5-nano` | ✅ ซื้อวัตถุดิบ 32,679.74 / ค่าใช้จ่ายบัตร 600 / ของใช้ส่วนตัว 252 | 76.1s | **3** | ✅ PASS |
| `gemini-2.5-flash-lite` | ✅ ตรงทั้ง 3 + รวม 33,531.74 | **16.3s** | 1 | ✅ PASS |
| `deepseek-v4-flash` | ✅ ตรงทั้ง 3 | 23.9s | 1 | ✅ PASS |

**สังเกต:** gpt-5-nano ใช้ **3 codeact steps** เพื่อหา top 3 (over-composition: น่าจะ list ทั้งหมดก่อน แล้วค่อย slice). gemini และ deepseek เรียก `sum_by_category` ครั้งเดียวแล้ว limit/slice ใน Python step เดียว → เร็วกว่า + ถูกกว่า

> **Anomaly:** deepseek run case 2 มี task argument จาก ReAct ที่ปนเปื้อน thinking tokens (เห็น `</suggestion></gadget>`, fake `Tool Response`, `<｜DSML｜tool_calls>` tags) **แต่ codeact ก็ยัง parse ออกมาถูกต้อง** — สะท้อนความ robust ของ codeact wrapper. นี่เป็นปัญหาที่ ReAct LLM (gpt-5-nano) — ไม่กระทบ scoring ของ codeact

### Case 3: "จ่ายให้ Starbucks ทั้งหมดเท่าไหร่"

| Model | Tool output | Latency | CodeAct steps | Score |
|---|---|---:|---:|:---:|
| `gpt-5-nano` | None / no record | 36.9s | 1 | ✅ PASS |
| `gemini-2.5-flash-lite` | None | 35.5s | 1 | ✅ PASS + insight แนะนำ alt spellings |
| `deepseek-v4-flash` | None | **12.9s** | 1 | ✅ PASS |

**สังเกตสำคัญ:**
- ทั้ง 3 model ตีความ "ทั้งหมด" เป็น ALL-TIME → ส่ง `start=2020-01-01, end=2026-05-17` (กฎ "ALL-TIME trigger" จาก system prompt)
- ใช้ `note_query='Starbucks'` ถูก (ไม่ confuse กับ category)
- **ไม่มี model ไหน hallucinate 310 THB** ที่ปรากฏใน raw DB query — ทุก model report None/0 ตรง ๆ ตาม wrapper. นี่คือ honesty test ที่ทุกตัวผ่าน
- gemini แสดง coaching ที่ดีสุด: เสนอ "Star" หรือชื่อเมนูแทน

### Case 4: "เปรียบเทียบรายจ่ายเดือนมีนากับเมษา แยกหมวด"

| Model | Tool output | Latency | CodeAct steps | Score |
|---|---|---:|---:|:---:|
| `gpt-5-nano` | ✅ March 6,350 vs April 33,909.74, diff +27,559.74 | 45.9s | 1 | ✅ PASS |
| `gemini-2.5-flash-lite` | ✅ ตรง + insight ดีสุด | **14.6s** | 1 | ✅ PASS |
| `deepseek-v4-flash` | ✅ ตรง + insight แนวทางธุรกิจ | 40.1s | 1 | ✅ PASS |

**สังเกต:** ทั้ง 3 ใช้ `compare_periods(by='category', period1_*, period2_*)` ในคำสั่งเดียว (ไม่ split). ตัวเลข diff/breakdown ตรงกัน 100%. gemini เร็วกว่า 3 เท่า

### Case 5: "กุมภาพันธ์ 2569 ใช้เงินกับอาหารทั้งหมดคิดเป็นบาทเท่าไร"

| Model | BE→AD | convert_to_thb | Tool output | Latency | CodeAct steps | Score |
|---|:---:|:---:|---|---:|---:|:---:|
| `gpt-5-nano` | ✅ 2569→2026 | ✅ | total_thb: 0 | 55.3s | **2** | ✅ PASS |
| `gemini-2.5-flash-lite` | ✅ | ⚠️ retry act_node 2× | amount_thb: None | 20.1s | 1+1 | ✅ PASS |
| `deepseek-v4-flash` | ✅ | ✅ | amount_thb: None | **11.8s** | 1 | ✅ PASS |

**สังเกตสำคัญ:**
- ทุก model **convert 2569→2026 ถูก** ใน task argument ("February 2026" ไม่ใช่ "February 2569") — นี่คือกฎที่ ReAct (gpt-5-nano) จัดการ ไม่ใช่ codeact
- gemini มี ReAct loop retry — ครั้งแรกส่ง task มี "converted to THB" ครั้งที่สองตัดออก (น่าจะคิดว่าผลแรกผิด). +1 act_node call = +cost
- gpt-5-nano ใช้ output key `total_thb` ตามที่ codeact ตัดสินใจ; gemini และ deepseek ใช้ `amount_thb` (ตรงกับ wrapper default)
- ไม่มี model ไหน hallucinate ยอด — ทุกตัว report None/0 ตรง ๆ ✅

---

## 4. Aggregate Scoreboard

### 4.1 Accuracy (จาก 5 cases × 1 คะแนน)

| Model | Case 1 | Case 2 | Case 3 | Case 4 | Case 5 | **Total** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `gpt-5-nano` | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| `gemini-2.5-flash-lite` | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| `deepseek-v4-flash` | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |

→ **ทุก model PASS ทุก case ในด้านความแม่นยำ** — wrapper เป็น source of truth และ codeact LLM ทำหน้าที่ "เลือก wrapper + ใส่ args ถูก" ได้ครบ

### 4.2 Latency (รวม 5 cases)

| Model | Total | Avg/case | Fastest case | Slowest case |
|---|---:|---:|---:|---:|
| `gpt-5-nano` | 243.9s | 48.8s | 29.7s (case 1) | 76.1s (case 2) |
| `gemini-2.5-flash-lite` | **101.7s** ⭐ | **20.3s** | 14.6s (case 4) | 35.5s (case 3) |
| `deepseek-v4-flash` | 112.0s | 22.4s | 11.8s (case 5) | 40.1s (case 4) |

→ **gemini เร็วที่สุดเฉลี่ย 2.4× กว่า gpt-5-nano**

### 4.3 CodeAct steps used (รวม 5 cases)

| Model | Total | Notes |
|---|---:|---|
| `gemini-2.5-flash-lite` | **5** ⭐ | 1 step ทุก case — most efficient composition |
| `deepseek-v4-flash` | 6 | over-iterate case 1 |
| `gpt-5-nano` | 8 | over-iterate case 2 (3 steps) + case 5 (2 steps) |

→ gemini compose ได้ในขั้นตอนเดียว, ไม่เปลือง wrapper call

### 4.4 Reply quality (subjective)

| Aspect | Best | Notes |
|---|---|---|
| Period framing | gpt-5-nano | "ครึ่งเดือนแรก" ใน case 1 — ตระหนัก partial month |
| Coaching insight | gemini | case 3: เสนอ alt spelling, case 4: highlight ที่ชัด |
| Visual layout | deepseek | ใช้ emoji + bullet ที่อ่านง่าย |
| Honesty (no hallucination) | ทุก model เสมอกัน | ทุกตัว report None/0 ตรงๆ ไม่มีตัวไหนแต่ง |

---

## 5. Verdict

### 🏆 อันดับ 1 — `google/gemini-2.5-flash-lite` (ปัจจุบัน)
- **Accuracy 5/5** เท่าทุก model
- **Latency เร็วที่สุด** เฉลี่ย 20.3s/case (2.4× กว่า gpt-5-nano)
- **CodeAct efficiency ดีที่สุด** — 1 step ทุก case
- Reply quality สูงกว่าค่าเฉลี่ย (coaching insights, alt-spelling suggestion)
- จุดอ่อนเดียว: case 5 ReAct retry act_node 2× — แต่ก็ยัง total เร็วกว่าตัวอื่น

### 🥈 อันดับ 2 — `deepseek/deepseek-v4-flash`
- **Accuracy 5/5**
- Latency #2 (22.4s/case)
- บางครั้ง over-iterate (case 1: 2 steps)
- Reply layout ดีที่สุดด้วย emoji + bullets

### 🥉 อันดับ 3 — `openai/gpt-5-nano`
- **Accuracy 5/5** (เสมอ)
- ⚠️ **Latency ช้าสุด 2.4× กว่า gemini** — เพราะ reasoning model มี hidden thinking tokens
- ⚠️ **Over-compose codeact steps** — case 2 ใช้ 3 steps (ควร 1), case 5 ใช้ 2 steps
- Period framing ดีที่สุด แต่ไม่คุ้ม cost/latency

---

## 6. Recommendation

**ไม่ต้องเปลี่ยน primary** — `google/gemini-2.5-flash-lite` ยังคงเป็น CodeAct model ที่เหมาะสมที่สุด

```env
CODEACT_MODEL=google/gemini-2.5-flash-lite      # KEEP (accuracy + speed + efficiency)
CODEACT_FALLBACK_MODELS=[
  "deepseek/deepseek-v4-flash",                  # KEEP (12 endpoints, code-strong)
  "openai/gpt-5-nano"                            # REPLACE old gpt-oss-safeguard fallback
]
```

**เปลี่ยนแปลงเทียบกับ config ปัจจุบัน:**
- **ลบ** `openai/gpt-oss-safeguard-20b:nitro` (safeguard model ไม่ใช่ coder + SPOF)
- **เพิ่ม** `openai/gpt-5-nano` เป็น fallback (ความแม่นยำเท่ากัน + cross-infra cover OpenAI/Azure)
- **กระจาย infra** 3 cloud: Google → DeepSeek/Parasail → OpenAI/Azure

### ผลคาดหวัง
- ราคา primary คงเดิม ($0.10/$0.40 per M)
- Fallback chain แข็งแกร่งขึ้น (ครอบคลุม 3 infra แทน 2 + ตัด safeguard ที่ผิดวัตถุประสงค์ออก)
- Accuracy ตามผล benchmark: ทุก fallback ก็ผ่าน 5/5 จึงไม่ degrade quality

---

## 7. Caveats + Limitations

1. **Sample size เล็ก** — 5 cases ต่อ model, n=1 per case. Run ซ้ำหลายรอบเพื่อ detect non-determinism (โดยเฉพาะ gpt-5-nano reasoning tokens)
2. **Ground truth จำกัด wallet ownership** — ข้อมูลของ user นี้มี orphan wallets เยอะ → cases 3+5 จึงเป็น "honesty test" ไม่ใช่ "math test". ถ้าข้อมูลครบ orphan filter, accuracy ranking อาจเปลี่ยน
3. **ReAct LLM = gpt-5-nano fixed** — variability ของ ReAct (task formatting, date computation, BE→AD) ส่งผลทั้ง 3 codeact runs เท่ากัน → control variable ดี. แต่ถ้าเปลี่ยน ReAct เป็น DeepSeek หรือ Gemini ภาพอาจต่าง
4. **Latency รวม ReAct + Act + CodeAct** — ไม่ได้แยก codeact-only time. การที่ gpt-5-nano ช้าอาจมาจาก reasoning tokens ในตัว codeact LLM (ไม่ใช่ ReAct)
5. **Cost ยังไม่ได้คำนวณจริง** — ดู `logs/codeact_bench_run.log` สำหรับ raw output; ต้อง pull จาก OpenRouter billing เพื่อ exact $/run
6. **No anti-hallucination negative test** — ทั้ง 3 cases ที่ wrapper return None, ทุก model honest. ควรเพิ่ม case ที่ wrapper return ตัวเลขเล็ก เพื่อทดสอบว่า model ขยายผลเอง (จาก context history) หรือไม่

---

## 8. Artifacts

- Runner script: `scripts/bench_codeact_accuracy.py`
- Smoke test: `scripts/bench_codeact_smoke.py`
- Raw results: `logs/codeact_bench_results.json` (15 runs × full tool_message + final_reply)
- Console log: `logs/codeact_bench_run.log`
- Ground truth queries: in this doc (section 1.2 + 2)

---

_Generated 2026-05-17. Ground truth fetched live via mcp__postgres__query. End-to-end model invocations via real OpenRouter API + LangGraph + PostgreSQL. Re-run if dataset, FX rates, or wrapper SQL change._
