# CodeAct Model Benchmark Report (2026-05-17)

**Target node:** `mint_agentic/src/graph/compute_subgraph/codeact/step.py` (`codeact_step_node`)
**Source of truth:** OpenRouter `/api/v1/models/<slug>/endpoints` (queried 2026-05-17)
**Goal:** เทียบ 11 candidate models เพื่อหา model ที่เหมาะกับ CodeAct loop ของ Mint Money

---

## 1. ความแตกต่างจาก ReAct node — สำคัญต่อการเลือก model

| ประเด็น | ReAct (`nodes.py`) | CodeAct (`step.py`) |
|---|---|---|
| Output | tool_call หรือ chat reply | **Python code block** (string) |
| Tool calling ที่ระดับ LLM API | **จำเป็น** | ไม่ต้อง — parse ด้วย `_extract_code()` |
| คุณภาพหลักที่ต้องการ | Decision + Thai NLU | **Code correctness + API contract compliance** |
| Loop | 1 turn → ตอบ | ลูป 1–5 step (ต่อ ReAct call 1 ครั้ง) |
| Model pool | ต้องมี `tools` ใน supported_parameters | กว้างกว่า — ใช้ model ไหนก็ได้ที่ออก Python ดี |

→ CodeAct ใช้ **coder-tuned models** ได้เต็มที่เพราะไม่ติดข้อจำกัด tool calling

---

## 2. Baseline ปัจจุบัน

```env
CODEACT_MODEL=google/gemini-2.5-flash-lite
CODEACT_BASE_URL=https://openrouter.ai/api/v1
CODEACT_FALLBACK_MODELS=["openai/gpt-oss-safeguard-20b:nitro","deepseek/deepseek-v4-flash"]
```

⚠️ **ปัญหาที่เจอในการตั้งค่าเดิม:** `openai/gpt-oss-safeguard-20b` เป็น **content-moderation/safeguard model** ที่ OpenAI ออกแบบสำหรับ classify policy violations ไม่ใช่ coder model. ที่แย่กว่าคือ **มี 1 endpoint = Groq เท่านั้น** → SPOF + ใช้ผิดวัตถุประสงค์ ควรเปลี่ยน

---

## 3. ความต้องการของ `codeact_step_node`

โหลด `step.py` แล้วสรุปได้ดังนี้:

1. **เขียน Python ตาม API contract ที่กำหนด** — ใช้ `resolve_wallet/category/tag()`, `sum_expense()`, `parse_period()`, `Decimal` ห้าม `float()`, ห้าม `import/exec/eval/getattr`
2. **System prompt ยาว ~10–15K tokens** — `_TOOL_REFERENCE` + `_INSTRUCTION` + entity catalog
3. **Multi-step loop (MAX_STEPS=5)** — model ต้องอ่าน `STDOUT/ERROR` ของ step ก่อนหน้าแล้วแก้ในรอบถัดไป
4. **ตัดสินใจเลือก wrapper ถูก** — `note_query` vs `category_names`, `convert_to_thb=True` vs manual sum, `clarify()` เมื่อกำกวม
5. **Thai phrase parsing** — `parse_period('เดือนที่แล้ว')`, BE→AD ผ่าน `parse_period('กุมภาพันธ์ 2569')`
6. **Cost-sensitive** — ทุก ReAct call เรียก 1–5 step → cost คูณ

---

## 4. Overview ทุก model (เรียงตาม cost/step ประมาณการ)

ประมาณการ cost: prompt 10K input + 1K output

| Model | Eps | Ctx | $in / $out (per M) | Est. $/step | Uptime 30m | จุดเด่น |
|---|---:|---:|---|---:|---|---|
| `openai/gpt-5-nano` | 3 | 400K | 0.05 / 0.40 | **$0.90** | 97–100% | ถูกสุด + reasoning + structured_outputs |
| `openai/gpt-oss-safeguard-20b` | **1** | 128K | 0.075 / 0.30 | $1.05 | 100% | 🚨 SPOF + ผิดวัตถุประสงค์ |
| `google/gemini-2.5-flash-lite` (ปัจจุบัน) | 3 | 1M | 0.10 / 0.40 | $1.40 | 99.6–99.9% | Google infra เสถียร |
| `deepseek/deepseek-v4-flash` | **12** | 1M | 0.14 / 0.28 | $1.68 | 98.4–99.9% | endpoint เยอะสุด, code เก่ง |
| `qwen/qwen3-coder-next` | 4 | 256K | 0.11–0.20 / 0.80–1.50 | $1.90–3.50 | 93–100% | **Coder-tuned โดยตรง** |
| `minimax/minimax-m2.5` | **17** | 200K | 0.15–0.60 / 0.59–2.40 | $2.10–4.00 | 77–100% (mix) | endpoint เยอะที่สุด |
| `inception/mercury-2` | **1** | 128K | 0.25 / 0.75 | $3.25 | 100% | 🚨 SPOF + diffusion-LLM |
| `minimax/minimax-m2.7` | 6 | 200K | 0.28–0.60 / 1.20–2.40 | $4.00–5.40 | 99–100% | รุ่นใหม่กว่า M2.5 แต่แพงขึ้น |
| `google/gemini-3.1-flash-lite` | 2 | 1M | 0.25 / 1.50 | $4.00 | 90–99% | แพงกว่า 2.5-lite 2.8 เท่า |
| `z-ai/glm-4.7` | 10 | 200K | 0.40–2.25 / 1.75–3.30 | $5.75–24.25 | 92–100% | code OK แต่แพง |
| `openai/gpt-5.3-codex` | 2 | 400K | 1.75 / 14.00 | **$31.50** | 99.8% | 🚨 codex flagship — overkill |

---

## 5. Production-Readiness Matrix

| Model | Endpoint risk | Python code quality | Thai parse | Cost vs ปัจจุบัน | สรุป |
|---|---|---|---|---|---|
| **gpt-5-nano** | 🟢 3 eps | 🟢 GPT-5 family | 🟢 | **-36%** | 🥇 Replace primary |
| **deepseek-v4-flash** | 🟢 12 eps | 🟢 DeepSeek เก่งโค้ด | 🟢 | +20% | 🥈 Safe quality upgrade |
| **qwen3-coder-next** | 🟢 4 eps (bf16 มี) | 🟢🟢 **Coder-tuned** | 🟡 Qwen Thai ใช้ได้ | +36% – +150% | 🥉 ดีสุดเฉพาะโค้ด แต่แพงขึ้น |
| **gemini-2.5-flash-lite** (ปัจจุบัน) | 🟢 | 🟡 OK ไม่เด่น | 🟢 | baseline | ✅ Keep เป็น fallback |
| **minimax-m2.5** | 🟢 17 eps | 🟢 code-strong | 🟡 | +50% – +185% | ⚠️ คุ้มเฉพาะ pin DeepInfra/Chutes |
| **gpt-oss-safeguard-20b** | 🔴 SPOF | 🔴 **ผิดวัตถุประสงค์** | 🟡 | -25% | ❌ ตัดออกจาก config |
| **mercury-2** | 🔴 SPOF | 🟡 ไม่มีหลักฐาน | 🔴 | +132% | ❌ ไม่เหมาะ |
| **gemini-3.1-flash-lite** | 🟡 2 eps | 🟡 | 🟢 | +185% | ❌ ไม่คุ้ม upgrade |
| **minimax-m2.7** | 🟢 6 eps | 🟢 | 🟡 | +185–285% | ❌ M2.5 ทำงานคุ้มกว่า |
| **glm-4.7** | 🟢 10 eps | 🟢 | 🟡 | +310%+ | ❌ แพงเกิน |
| **gpt-5.3-codex** | 🟢 2 eps | 🟢🟢 best-in-class | 🟢 | **+2150%** | ❌ overkill มหาศาล |

---

## 6. Verdict

### 🥇 อันดับ 1 — `openai/gpt-5-nano` (challenger หลัก)

**เหตุผล:**
- **ราคาถูกกว่าปัจจุบัน 36%** ที่ input — codeact loop 1–5 step → ประหยัดทบเท่าต่อ ReAct call
- มี `reasoning` parameter → ช่วย multi-step composition (decide → compose → verify)
- รองรับ `structured_outputs` (เผื่ออนาคต)
- GPT-5 family Python literacy สูง — compose pre-defined wrappers สบาย
- Context 400K — system prompt + 5 step history ใส่ได้สบายมาก
- 3 endpoints บน 2 infra (OpenAI + Azure x2)

**ข้อควรระวัง:**
- เป็น reasoning model → output billing รวม hidden thinking tokens
- ต้อง benchmark API contract compliance ก่อน promote

### 🥈 อันดับ 2 — `deepseek/deepseek-v4-flash`

**เหตุผล:**
- **12 endpoints** = redundancy ดีสุดในกลุ่ม
- DeepSeek เป็นที่รู้กันว่าเก่งโค้ดที่ราคา flash
- 1M context, auto prefix-caching ฟรี
- ทดสอบมาแล้วในโปรดักชัน (อยู่ใน fallback list ตอนนี้)

**ข้อควรระวัง:**
- แพงกว่าปัจจุบัน 20% — คุ้มเฉพาะถ้า quality เพิ่มชัด
- DeepInfra เป็น fp4 quality drop — pin endpoint ถ้าจะใช้

### 🥉 อันดับ 3 (เฉพาะถ้า codeact error rate สูง) — `qwen/qwen3-coder-next`

**เหตุผล:**
- **Coder-tuned โดยตรง** — เก่งสุดด้านโค้ดในลิสต์นี้
- Parasail bf16 = ไม่มี quantization drop
- 4 endpoints

**ข้อควรระวัง:**
- ราคาขึ้น 36–150% — คุ้มเฉพาะถ้า benchmark ชี้ว่า code-quality สำคัญกว่า cost
- Qwen Thai parse ผ่าน `parse_period()` ต้องทดสอบ

---

## 7. Recommended Config

```env
CODEACT_MODEL=openai/gpt-5-nano
CODEACT_FALLBACK_MODELS=[
  "deepseek/deepseek-v4-flash",
  "google/gemini-2.5-flash-lite"
]
```

**เหตุผล:**
- ครอบคลุม 3 cloud infrastructure (OpenAI/Azure, DeepSeek+OSS providers, Google)
- **ลบ `openai/gpt-oss-safeguard-20b:nitro` ออก** — safeguard model ผิดวัตถุประสงค์ + SPOF (Groq เท่านั้น)
- Primary ราคาลง 36%, fallback แรกราคาขึ้น 20% แต่ quality เพิ่มชัด
- Fallback เรียงตามคุณภาพโค้ด: gpt-5-nano → deepseek (high quality, eps เยอะ) → gemini-lite (เสถียร)

---

## 8. ❌ ตัดออกชัดเจน

| Model | เหตุผล |
|---|---|
| `openai/gpt-oss-safeguard-20b` | safeguard ≠ coder + SPOF (Groq เท่านั้น) — เลิกใช้ใน fallback ปัจจุบัน |
| `inception/mercury-2` | SPOF + diffusion-LLM ยังไม่มีหลักฐาน code benchmark |
| `google/gemini-3.1-flash-lite` | แพงกว่า 2.5-lite 2.8x แต่ flash-lite tier upgrade เล็ก |
| `openai/gpt-5.3-codex` | $1.75/$14 — overkill มหาศาล สำหรับ compose-wrapper task |
| `minimax/minimax-m2.7` | ใหม่กว่า M2.5 แต่ราคาขึ้น 100%+ ไม่คุ้ม |
| `z-ai/glm-4.7` | code OK แต่ราคา/eps ไม่คุ้มเทียบ qwen-coder |

---

## 9. Caveats ก่อน ship gpt-5-nano เป็น primary

### Test cases ที่ต้อง benchmark (20+ scenarios)

1. **Resolve flow**
   - `resolve_wallet('true money')` → 'TrueMonney'
   - `resolve_category('เดินทาง')` → expansion เป็น list ของ subcategories
   - `resolve_tag('#ประชุมงาน')` preserve `#` prefix
   - Handle `AmbiguousMatchError` ในรอบถัดไป

2. **Time handling**
   - ISO dates verbatim — `start = 2026-02-05, end = 2026-05-05` ต้อง map เป็น `date(2026, 2, 5)` ไม่ re-parse
   - Window vs single-point — "over the last 3 months window" ≠ "for last month (single calendar month)"
   - `parse_period('เดือนที่แล้ว')`
   - BE→AD ผ่าน `parse_period('กุมภาพันธ์ 2569')`

3. **Wrapper selection**
   - `note_query` (vendor: Starbucks, Cafe Amazon) vs `category_names` (หมวด)
   - `convert_to_thb=True` แทนการ manual sum across currencies
   - `compare_periods()` แทน manual diff
   - `spending_trend(group_by='month')` แทน loop 6 ครั้ง

4. **Error recovery**
   - Step 1 raises `ValueError("no wallet matches 'savings'. Available: [...]")` → Step 2 อ่าน Available list แล้วเลือกถูก
   - `AmbiguousMatchError` → เรียก `clarify()` ถูกจังหวะ

5. **Constraints compliance**
   - ห้าม `float()` กับเงิน
   - ห้าม `import`, `open`, `exec`, `eval`, `getattr`, dunder access
   - ห้าม manual sum cross-currency
   - ห้ามทำเกิน MAX_STEPS=5

### Operational metrics ต้องวัด

1. **Per-step latency p50/p95/p99** — codeact loop 5 step = latency คูณ 5
2. **Average steps to completion** — ถ้า gpt-5-nano ใช้ 3 step ตอบ vs gemini ใช้ 2 → cost จริงอาจไม่ดีกว่า
3. **Error rate ต่อ step** — sandbox `ValueError`, `AmbiguousMatchError`, syntax errors
4. **Reasoning token billing actual** — เทียบ predicted vs actual
5. **Cache hit rate** — ทั้ง gpt-5-nano และ deepseek auto-cache prefix

---

## 10. Action Items

- [ ] รัน A/B test: gemini-2.5-flash-lite vs gpt-5-nano vs deepseek-v4-flash บน 20 test cases ข้างบน
- [ ] วัด avg steps + p95 latency จาก compute_subgraph log
- [ ] เทียบ effective cost per ReAct call (รวม cache hit + reasoning tokens)
- [ ] ลบ `openai/gpt-oss-safeguard-20b:nitro` ออกจาก CODEACT_FALLBACK_MODELS — ไม่ว่าจะตัดสินใจอย่างไรเรื่อง primary
- [ ] ถ้า gpt-5-nano ผ่าน → swap primary + ปรับ fallback chain ตามข้อ 7
- [ ] ถ้า code error rate ยังสูง → ทดลอง qwen3-coder-next pin `provider.only=["Parasail"]` (bf16)

---

_Generated by `openrouter-analyzer` skill — data pulled live from OpenRouter `/api/v1/models/<slug>/endpoints` on 2026-05-17. Re-run analysis if > 2 weeks old since provider routing and pricing change frequently._
