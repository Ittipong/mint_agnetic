# สรุป Benchmark: เลือกโมเดลสำหรับ Quick Add Node

**วันที่:** 2026-05-17
**Target node:** `quick_add_node` (text → ask-back หรือ `propose_transaction` tool call)
**Bench script:** `mint_agentic/test_quick_add_bench_multi.py`
**ชุดทดสอบ:** 10 cases ภาษาไทยจริง ครอบคลุม single-turn complete add, multi-turn merge, ask-back, wallet match, income detection, currency lock

## ทำไมต้อง bench

`quick_add_node` เป็น text-only counterpart ของ slip_node — ทำงาน 2 mode:

1. **Tool-fire mode** — มี `name + amount` ครบ → call `propose_transaction` ทันที (single-turn complete)
2. **Ask-back mode** — ขาด required field → ตอบเป็น Thai sentence ขอข้อมูลเพิ่ม

โมเดลที่ใช้กับ node นี้ต้องผ่าน 3 ด่าน:

- **Tool calling reliability** — fire `propose_transaction` เมื่อควร fire / ตอบ text เมื่อควรถาม
- **Argument correctness** — `amount`, `type` (expense/income), `note`, `wallet_id` ถูกต้อง
- **Multi-turn context merging** — รวม "เที่ยว" (turn 1) + "200" (turn 2) เป็น transaction เดียว

อยู่บน hot path เช่นเดียวกับ intent classifier — latency สำคัญ

## รายชื่อ candidates

| Model | Endpoints | Provider HA | tools | structured_outputs | $/M (in / out) | Pre-filter |
|---|---|---|---|---|---|---|
| `inception/mercury-2` | 1 | Inception (UNKNOWN) | ✅ | ✅ | $0.25 / $0.75 | ผ่าน |
| `mistralai/devstral-small` | 1 | Mistral-direct | ✅ | ✅ | $0.10 / $0.30 | ผ่าน |
| `nvidia/nemotron-nano-9b-v2` | 1 | DeepInfra (HIGH-RPM) | ✅ | ❌ | $0.04 / $0.16 | ผ่าน |
| `openai/gpt-5-nano` | **3** | Azure ×2 + OpenAI | ✅ | ✅ | $0.05 / $0.40 | ผ่าน |
| `google/gemini-2.5-flash-lite` | **3** | Vertex ×2 + AI Studio | ✅ | ✅ | $0.10 / $0.40 + cache $0.01 | ผ่าน |
| `deepseek/deepseek-v4-flash` | **12** | DeepSeek/Baidu/SiliconFlow/etc. | ✅ (most) | ✅ (some) | $0.14 / $0.28 | ผ่าน |
| `qwen/qwen3-coder-30b-a3b-instruct` | 4 | Novita/SiliconFlow/Bedrock/Alibaba | ✅ | ✅ (some) | $0.07 / $0.27 | ผ่าน |
| ~~`meta-llama/llama-3.2-1b-instruct`~~ | 1 | Cloudflare | **❌** | ❌ | $0.027 / $0.20 | **ตัดออก — ไม่มี tools** |
| ~~`mistralai/mistral-small-24b-instruct-2501`~~ | 1 | DeepInfra | **❌** | ✅ | $0.05 / $0.08 | **ตัดออก — ไม่มี tools** |

แหล่งข้อมูล: `https://openrouter.ai/api/v1/models/<slug>/endpoints` (ดึง 2026-05-17)

### Pre-filter logic

`quick_add_node` ต้องใช้ tool-calling ผ่าน `propose_transaction` — endpoint ที่ `tools=false` fire tool ไม่ได้ → ใช้ไม่ได้:

- `meta-llama/llama-3.2-1b-instruct` (Cloudflare) ตัดออก
- `mistralai/mistral-small-24b-instruct-2501` (DeepInfra) ตัดออก

เหลือ **7 candidates** ที่จะ bench จริง

## ผล Benchmark

| Model | Pass | p50 | p95 | avg |
|---|---|---|---|---|
| `inception/mercury-2` | **10/10 (100%)** | **1401ms** | 2001ms | 1417ms |
| `openai/gpt-5-nano` | 10/10 (100%) | 17148ms | 25338ms | 14945ms |
| `deepseek/deepseek-v4-flash` | 9/10 (90%) | 5977ms | 8837ms | 6210ms |
| `mistralai/devstral-small` | 8/10 (80%) | 1391ms | 2149ms | 1430ms |
| `google/gemini-2.5-flash-lite` | 8/10 (80%) | 2231ms | 4239ms | 2480ms |
| `qwen/qwen3-coder-30b-a3b-instruct` | 7/10 (70%) | 1975ms | 3089ms | 2222ms |
| `nvidia/nemotron-nano-9b-v2` | 6/10 (60%) | 4257ms | 5328ms | 4358ms |

### รายละเอียดเคสที่พลาด

**`inception/mercury-2`** — ผ่านครบ 10/10 ✅

**`openai/gpt-5-nano`** — accuracy 100% แต่ **p50 = 17 วินาที** → ใช้ใน hot path ไม่ได้

**`deepseek/deepseek-v4-flash`** — พลาด 1 case:
- `"โอน 500 ให้แม่"` → ตอบ empty text + ไม่ fire tool

**`mistralai/devstral-small`** — พลาด 2 cases (จับ amount ไม่ได้บางรูปแบบ):
- `"ซื้อหนังสือ $20"` → ask-back ทั้งที่มี $20 (treat $20 ไม่ใช่ amount)
- `"โอน 500 ให้แม่"` → ask-back ทั้งที่มี 500

**`google/gemini-2.5-flash-lite`** — พลาด 2 cases เป็น **empty-response bug** ⚠️
- `"กิน kfc 100"` → text="" + no tool_call (model glitch)
- `"จ่ายค่าไฟ 1200 จาก KBank"` → text="" + no tool_call
- เป็น production-blocker bug ที่ต้องอาศัย defensive recovery รับมือ

**`qwen/qwen3-coder-30b-a3b-instruct`** — พลาด 3 cases (ask-back ทั้งที่ amount อยู่ในข้อความ):
- `"ซื้อกาแฟ 65 บาท"`, `"ซื้อหนังสือ $20"`, `"โอน 500 ให้แม่"`
- Coding-specialist → Thai verb-noun-amount parsing อ่อน

**`nvidia/nemotron-nano-9b-v2`** — พลาด 4 cases:
- 3 cases: fire tool ได้แต่ `note` field ว่าง (ไม่ copy ชื่อสินค้าลง note)
- 1 case: `"โอน 500 ให้แม่"` → ask-back

## ข้อสังเกตสำคัญ

### 1. Mercury-2 ครองแชมป์อีกครั้ง

ตรงกับผล bench `intent_classifier` ก่อนหน้านี้ Mercury-2 ทำได้ทั้ง accuracy 100% + p50 อยู่ในกรอบที่ยอมรับได้ของ hot path

### 2. HA paradox

โมเดลที่มี endpoint redundancy ดีที่สุดล้วน **ใช้ไม่ได้บน hot path**:

- `gpt-5-nano` (3 endpoints — Azure×2 + OpenAI) → p50 17s ช้าเกิน
- `gemini-2.5-flash-lite` (3 endpoints — Vertex×2 + AI Studio) → empty-response bug ~20% rate
- `deepseek-v4-flash` (12 endpoints — มากที่สุด) → p50 6s ช้าเกิน

ตัวที่เร็วและ accuracy ดีกลับเป็น single endpoint ทั้งคู่ (Mercury, Devstral) → ทำให้ต้อง pair primary + fallback คนละ infra

### 3. Pareto frontier มี 2 จุด

จุดที่ accuracy + latency ดีพร้อมกัน:

- Mercury-2 (100%, 1.4s, single endpoint)
- Devstral-small (80%, 1.4s, single endpoint)

ตัวอื่นแย่ใน axis ใด axis หนึ่ง — ไม่อยู่บน frontier

### 4. Coding/Reasoning model ไม่เหมาะ

- `qwen3-coder-30b` (coding model) → Thai chat understanding อ่อน, ask-back ผิดเวลา
- `nemotron-nano-9b-v2` (reasoning model) → drop `note` field + ช้าเพราะ hidden thinking tokens

Specialist model สำหรับงาน chat tool-calling ทั่วไป = mismatch

### 5. Empty-response bug ของ Gemini

เป็น failure mode ที่ผิดปกติ — model ตอบกลับมาเป็น `text=""` ไม่มี tool_call ในเคสที่ควรชัดเจน (มี amount + verb ครบ) คาดว่าเป็น quirk ของ Gemini ผ่าน OpenRouter ทำให้แม้ HA ดีก็ยังเสี่ยง

โค้ดปัจจุบันใน `quick_add_node.py` มี `_recover_tool_args_from_text()` (defensive recovery) อยู่แล้ว แต่ recover จาก JSON/Python-call text ไม่ใช่จาก empty response → ถ้าใช้ Gemini ต้องเพิ่ม retry-on-empty หรือ swap to fallback ทันที

## คำแนะนำ

### Option A (แนะนำ) — Quality first + cross-infra fallback

```
primary:   inception/mercury-2          (Inception, 100% accuracy)
fallback:  google/gemini-2.5-flash-lite (Google Vertex/AI Studio — ต่าง infra)
```

- ครอบ SPOF ของ Inception ด้วย Google 3 endpoints HA
- ต้องเสริม retry-on-empty / fallback-on-empty สำหรับ Gemini quirk
- Cost: $0.25/$0.75 primary, $0.10/$0.40 fallback — Gemini ถูกกว่าครึ่ง

### Option B — Cost + HA first, ยอม accuracy ลด 20%

```
primary:   google/gemini-2.5-flash-lite (3 endpoints + auto-cache)
fallback:  inception/mercury-2          (quality safety net)
```

- ราคาถูกกว่า + auto-caching ($0.01/M cache read)
- Empty-response 20% rate รับด้วย Mercury fallback → accuracy รวมน่าจะใกล้ 100%
- ต้องวัด end-to-end accuracy หลัง fallback chain ก่อน ship

### ตัวที่ตัดออก

- `openai/gpt-5-nano` — accuracy 100% แต่ p50 17s = ผู้ใช้ปิดแอปก่อนเห็นการ์ด
- `deepseek/deepseek-v4-flash` — p50 6s ช้าเกิน, hidden thinking tokens
- `mistralai/devstral-small` — เร็วแต่ accuracy 80%, จับ amount บางรูปแบบไม่ได้
- `qwen/qwen3-coder-30b` — coding-specialist, ไม่เหมาะ Thai chat
- `nvidia/nemotron-nano-9b-v2` — drop `note` field + reasoning latency
- `meta-llama/llama-3.2-1b-instruct` — ไม่มี tools (Cloudflare endpoint)
- `mistralai/mistral-small-24b-instruct-2501` — ไม่มี tools (DeepInfra endpoint)

## ข้อควรระวังก่อน ship

1. **10 cases ยังน้อย** — ก่อน commit จริงควรขยายเป็น 50+ cases ครอบคลุม:
   - หลายสกุล wallet (KBank, SCB, TrueMoney, cash) — ทดสอบ wallet matching cascade
   - หลาย category (อาหาร, ค่าใช้จ่ายบ้าน, ช้อปปิ้ง, etc.) — ทดสอบ category matching
   - Edge cases: amount แบบ 1.5K / "พันห้า" / "หนึ่งหมื่นห้า"
   - การกด edit หลังการ์ดโผล่ (`_looks_like_edit` flow)
2. **ไม่ได้ทดสอบ `corrects_group_id` flow** — bench นี้สมมุติว่าไม่มี pending card ทุก case
3. **Catalog เป็น mock** — production catalog ของผู้ใช้จริงจะมี wallet/category เยอะกว่า → prompt ใหญ่ขึ้น → latency เพิ่ม → ต้อง re-bench หลังเก็บ real prompt size
4. **Cache savings ไม่ได้ model** — system prompt ของ quick_add ใหญ่ + static → Gemini auto-cache จะลด cost จริงในการใช้งาน, Mercury ไม่มี cache → จะแพงกว่า Gemini หลายเท่าเมื่อใช้จริง
5. **Currency-lock test ผ่าน แต่ยังแคบ** — โค้ดมี force-set currency หลัง LLM ทำให้ทดสอบนี้ไม่ critical แต่ถ้า LLM พลาดด้วยจะดูเหมือนยอม "$20" → ทดสอบ end-to-end จริงควรเช็คว่า args ที่ส่งไป mobile เป็น THB/฿ เสมอ
6. **Provider variance** — DeepSeek v4 Flash มี 12 endpoints quantization ต่างกัน (fp4/fp8/unknown) → ผลอาจต่างถ้า OpenRouter route ไป provider อื่น
7. **Coverage ของ multi-turn ยังตื้น** — มีแค่ 2 turn 2 cases → ต้องเพิ่ม 3+ turn scenarios เช่น "เที่ยว" → "ที่ภูเก็ต" → "200"

## วิธี Reproduce

```bash
cd mint_agentic
uv run python test_quick_add_bench_multi.py
```

แก้รายการ `MODELS` ใน script ได้ตามต้องการ รันครั้งหนึ่งใช้เวลา ~5 นาที สำหรับ 7 โมเดล × 10 cases (gpt-5-nano อย่างเดียวก็กิน 2.5 นาที)
