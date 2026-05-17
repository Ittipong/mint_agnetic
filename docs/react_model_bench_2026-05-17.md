# ReAct Agent — Model Benchmark Report (2026-05-17)

**Target node:** `mint_agentic/src/graph/nodes.py` (`reason_node`)
**Source of truth:** OpenRouter `/api/v1/models/<slug>/endpoints` (queried 2026-05-17)
**Goal:** เทียบ 9 candidate models กับ baseline ปัจจุบัน เพื่อหา model ที่เหมาะกับ ReAct chat agent ของ Mint Money

---

## 1. Baseline ปัจจุบัน

```env
REACT_MODEL=deepseek/deepseek-v4-flash
REACT_BASE_URL=https://openrouter.ai/api/v1
REACT_FALLBACK_MODELS=["google/gemini-2.5-flash-lite","qwen/qwen3.6-35b-a3b"]
```

**ความต้องการของ `reason_node`:**
- Tool calling จำเป็น (3 tools: `analyze_user_finances`, `get_financial_advice`, `lookup_entity`)
- System prompt ยาวประมาณ 10–15K tokens (slip context + entity catalog + rules ทั้งหมด)
- History budget 4K tokens หลัง compression (`_HISTORY_TOKEN_BUDGET`)
- ตอบเป็นภาษาไทย, ทำตาม long-instruction prompt ที่ซับซ้อน (window vs single-point, BE→AD, note-search routing ฯลฯ)
- Hot path — ทุก turn ของ chat เรียก node นี้ → ต้องถูก + เร็ว + เสถียร
- ต้องการ structured output สำหรับ `<suggestions>` tag

---

## 2. Overview ทุก model (ราคา $/M tokens)

| Model | Eps | Tools | Ctx | $in / $out | Uptime 30m | Note |
|---|---:|:---:|---:|---|---|---|
| `deepseek/deepseek-v4-flash` ✅ปัจจุบัน | **12** | ✓ ส่วนใหญ่ | 1M | 0.14 / 0.28 | 99.4–99.9% | DeepSeek+Parasail+Novita+AtlasCloud คุณภาพดี, DeepInfra เป็น fp4 ระวัง |
| `google/gemini-2.5-flash-lite` ✅fallback | 3 | ✓ ทุก ep | 1M | 0.10 / 0.40 | 99.5–99.9% | Google-only (region redundancy), structured_outputs ✓ |
| `openai/gpt-5-nano` 🌟 | 3 | ✓ ทุก ep | 400K | **0.05 / 0.40** | 97.7–99.96% | **ถูกที่สุด** + reasoning + structured_outputs, OpenAI+Azure |
| `openai/gpt-oss-120b` | **18** | ✓ ส่วนใหญ่ (Bedrock/Google ไม่มี) | 128K | 0.04–0.35 / 0.19–0.75 | 78–100% (mix) | endpoint เยอะที่สุด แต่ quantization ผสม fp4/fp8/bf16 → คุณภาพแกว่ง |
| `stepfun/step-3.5-flash` | 4 | ✓ ทุก ep | 256K | 0.10 / 0.30 | 97.5–100% | จีน-origin, **Thai support unknown ต้อง benchmark** |
| `google/gemma-4-31b-it` | 7 | ⚠️ บางเจ้าไม่มี (DeepInfra-fp4, Chutes) | 256K | 0.12–0.20 / 0.37–0.50 | 99.2–99.9% | Gemma เด่น instruction-follow แต่ tool dialect ไม่แน่นอน |
| `mistralai/devstral-small` | **1** | ✓ | 128K | 0.10 / 0.30 | 100% | 🚨 **SPOF** + code-tuned ไม่ใช่ chat, Thai อ่อน |
| `inception/mercury-2` | **1** | ✓ + structured_outputs | 128K | 0.25 / 0.75 | 100% | 🚨 **SPOF** + diffusion-LLM แปลกใหม่ |
| `openai/gpt-4o-mini` | 3 | ⚠️ **เฉพาะ OpenAI direct** | 128K | 0.15 / 0.60 | 99.7–99.99% | 🚨 Azure 2 endpoints **ไม่มี `tools` parameter** — route ไปเจอ 400 |

---

## 3. Production-Readiness Matrix

| Model | Endpoint Risk | Tool support ทั่วถึง | Thai quality | Cost vs ปัจจุบัน | สรุป |
|---|---|---|---|---|---|
| **gpt-5-nano** | 🟢 3 eps | 🟢 ทุก ep | 🟢 OpenAI Thai tier-1 | **-64% in / +43% out** | 🥇 Replace candidate |
| **deepseek-v4-flash** (ปัจจุบัน) | 🟢 12 eps | 🟢 | 🟢 ทดสอบมาแล้ว | baseline | ✅ Keep — ยังคุ้ม |
| **gemini-2.5-flash-lite** | 🟢 3 eps | 🟢 | 🟢 | -29% in / +43% out | 🥈 Keep เป็น fallback |
| **gpt-oss-120b** | 🟢 endpoint แต่ 🟡 quant | 🟡 (Bedrock/Google ตัด tools) | 🟡 OSS-tier | -71% in / -32% out @DeepInfra | ⚠️ ดูคุ้ม แต่ต้อง pin provider + benchmark |
| **gemma-4-31b-it** | 🟢 7 eps | 🟡 (ต้อง pin provider) | 🟡 พอใช้ | -14% in / +35% out | ⚠️ ไม่คุ้มเสี่ยง |
| **step-3.5-flash** | 🟢 4 eps | 🟢 | 🔴 **unknown** | -29% in / +7% out | ❌ ไม่ผ่านเกณฑ์ภาษา |
| **devstral-small** | 🔴 **SPOF** | 🟢 | 🔴 code-tuned | -29% in / +7% out | ❌ ไม่เหมาะ ReAct chat |
| **mercury-2** | 🔴 **SPOF** | 🟢 | 🔴 ไม่มีข้อมูล | +79% in / +168% out | ❌ ไม่เหมาะ production |
| **gpt-4o-mini** | 🔴 **routing risk** | 🔴 (Azure ไม่มี tools) | 🟢 | +7% in / +114% out | ❌ ต้อง pin OpenAI-only → 1 ep + แพงกว่า |

**หมายเหตุ "Azure ไม่มี tools":** จาก `/endpoints` response ของ `openai/gpt-4o-mini` — Azure rows มี `supported_parameters` ที่ขาด `tools` และ `tool_choice`; เฉพาะ OpenAI direct เท่านั้นที่รองรับ. OpenRouter อาจ auto-skip แต่จะลด pool การ route และเสี่ยง 400 ถ้าไม่ pin

---

## 4. Verdict

### 🥇 อันดับ 1 — `openai/gpt-5-nano` (challenger ควรทดลอง)
- **ราคา in $0.05/M ถูกกว่า DeepSeek 64%** — system prompt 10–15K tokens ส่งทุก turn → ประหยัดมาก
- มี reasoning + structured_outputs (เหมาะกับ `<suggestions>` JSON)
- ภาษาไทย tier-1 ของ OpenAI
- Context 400K เกินพอ
- ⚠️ ต้อง benchmark p99 latency + long-instruction following กับ prompt จริงก่อน promote

### ✅ คงไว้ — `deepseek/deepseek-v4-flash` (primary)
- 12 endpoints = SPOF risk ต่ำสุดในกลุ่ม
- ทดสอบ Thai มาแล้วในโปรดักชัน
- Auto prefix caching ช่วยลดต้นทุน system prompt
- Context 1M

### 🥈 Recommended Fallback Chain (cross-infra)

```env
REACT_MODEL=deepseek/deepseek-v4-flash         # DeepSeek+Parasail+Novita
REACT_FALLBACK_MODELS=[
  "google/gemini-2.5-flash-lite",              # Google infra (อัพเดต)
  "openai/gpt-5-nano"                          # OpenAI/Azure infra (ใหม่)
]
```

**เหตุผล:** ครอบคลุม 3 cloud (DeepSeek+ผู้ให้บริการ OSS, Google, OpenAI/Azure) — โอกาสที่ทั้ง 3 ล่มพร้อมกันต่ำมาก. ตัด `qwen/qwen3.6-35b-a3b` ออกเพราะคู่ DeepSeek+Gemini+OpenAI หลากหลายกว่าอยู่แล้ว

### ❌ ตัดออกชัดเจน

| Model | เหตุผล |
|---|---|
| `mercury-2` | Single-provider, diffusion-LLM ทดลอง, แพงสุด, Thai unknown |
| `devstral-small` | Single-provider, code-tuned ไม่ใช่ chat, Thai อ่อน |
| `gpt-4o-mini` | Azure endpoints ไม่รองรับ `tools` → ReAct จะ 400 เป็นช่วงๆ จนกว่าจะ pin OpenAI-only (เสีย redundancy + แพงกว่า nano 3 เท่า) |
| `step-3.5-flash` | Thai support ไม่มีหลักฐาน, จีน-origin |
| `gemma-4-31b-it` | Tool support ไม่สม่ำเสมอข้าม provider, fp4 quant คุณภาพแกว่ง |
| `gpt-oss-120b` | คุ้มราคาแต่ Bedrock/Google providers ไม่มี tools, fp4 หลายเจ้า → ต้อง `provider.only` ถึงจะปลอดภัย; ใช้ได้ในงาน non-critical แต่ไม่ใช่ ReAct chat |

---

## 5. Caveats — ต้อง benchmark ก่อน ship gpt-5-nano

1. **Real-world Thai benchmark** — รัน `build_system_prompt` จริงกับ test cases ตัวแทน:
   - Slip intent (`[INTENT:parse_transaction_from_slip]`)
   - ALL-TIME trigger ("ทั้งหมด", "เคย...", "กี่ครั้งแล้ว")
   - Credit card routing (`type: creditcard`)
   - Multi-currency conversion ("คิดเป็นบาท")
   - Note search vs category disambiguation (Starbucks, Bolt, ฯลฯ)
   - Window vs single-point ("3 เดือนที่แล้ว" — should default to window)
   - BE→AD year conversion ("2569" → "2026")
   - `<suggestions>` JSON format compliance
2. **p99 latency under load** — `/endpoints` ไม่บอก; วัดจาก SSE log จริง
3. **Tool-call schema** — gpt-5-nano ใช้ OpenAI format เดียวกับ DeepSeek → สลับได้ตรงๆ
4. **Cache hit rate** — DeepSeek auto-cache prefix ฟรี, gpt-5-nano auto-cache เช่นกัน แต่อัตราต่างกัน → คำนวณ effective cost จาก real cache hit rate ก่อนเทียบ headline price
5. **Reasoning token billing** — gpt-5-nano เป็น reasoning model → output billing รวม reasoning tokens (ไม่เห็นใน response). ดู `pricing.internal_reasoning` ก่อนคำนวณงบจริง
6. **Provider pinning** — ถ้า OpenAI endpoint ของ gpt-5-nano uptime แกว่ง (97.7% เห็นในตอนนี้) อาจต้อง `provider.order=["Azure", "OpenAI"]`

---

## 6. Action Items

- [ ] รัน A/B test: DeepSeek-v4-flash vs gpt-5-nano บน test prompts ชุดข้างบน
- [ ] วัด p50/p95/p99 latency จาก SyncFileLogger / reason_debug log
- [ ] เทียบ effective cost ต่อ 1,000 turns (รวม cache hit rate จริง + reasoning tokens สำหรับ nano)
- [ ] ถ้า nano ผ่าน → ปรับ `REACT_FALLBACK_MODELS` ใช้ chain ข้างบน
- [ ] ถ้า nano ไม่ผ่าน → keep config ปัจจุบัน + แทน `qwen3.6-35b-a3b` ด้วย `openai/gpt-5-nano` เพื่อ cross-infra fallback

---

_Generated by `openrouter-analyzer` skill — data pulled live from OpenRouter `/api/v1/models/<slug>/endpoints` on 2026-05-17. Re-run analysis if > 2 weeks old since provider routing and pricing change frequently._
