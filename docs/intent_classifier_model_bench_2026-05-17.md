# สรุป Benchmark: เลือกโมเดลสำหรับ Intent Classifier

**วันที่:** 2026-05-17
**Target node:** `classify_intent_node` (entry-router ที่รันทุก text turn)
**Bench script:** `mint_agentic/test_intent_bench_multi.py`
**ชุดทดสอบ:** 25 cases ภาษาไทยจริง ครอบคลุม add_transaction (มียอด / bare noun / follow-up / edit), analytics, chit-chat, adversarial

## ทำไมต้อง bench

`classify_intent_node` ตัดสินใจระหว่าง **add_transaction** (เข้า quick-add lane) กับ **other** (เข้า ReAct lane) ทุก turn ที่ user พิมพ์ ถ้าเลือกผิด:

- **Analytics → add_transaction** = โผล่ proposal card ที่ไม่ควรโผล่ (UX แย่)
- **Follow-up amount → other** = quick-add flow พัง user ต้องพิมพ์ซ้ำ

อยู่บน hot path ด้วย → **latency รู้สึกได้ทุก turn**

Bench นี้เทียบ 8 โมเดล candidate วัด **accuracy + p50/p95 latency** บนชุดเคสจริง + ดึง `/endpoints` API ของ OpenRouter เพื่อยืนยัน provider redundancy และ supported parameters

## รายชื่อ candidates

| Model | Endpoints | Provider class | tools | tool_choice | response_format | structured_outputs | $/M (in / out) |
|---|---|---|---|---|---|---|---|
| `inception/mercury-2` | 1 | Inception (UNKNOWN) | ✅ | ✅ | ✅ | ✅ | $0.25 / $0.75 |
| `mistralai/devstral-small` | 1 | Mistral-direct | ✅ | ✅ | ✅ | ✅ | $0.10 / $0.30 |
| `nvidia/nemotron-nano-9b-v2` | 1 | DeepInfra (HIGH-RPM) | ✅ | ✅ | ✅ | ❌ | $0.04 / $0.16 |
| `openai/gpt-5-nano` | **3** | Azure ×2 + OpenAI | ✅ | ✅ | ✅ | ✅ | $0.05 / $0.40 |
| `amazon/nova-lite-v1` *(ปัจจุบัน)* | **2** | Bedrock (HIGH-RPM) | ✅ | ❌ | ❌ | ❌ | $0.06 / $0.24 |
| `ibm-granite/granite-4.1-8b` | 1 | WandB (UNKNOWN) | ✅ | ✅ | ✅ | ✅ | $0.05 / $0.10 |
| `openai/gpt-oss-safeguard-20b` | 1 | Groq (LOW-RPM) | ✅ | ✅ | ✅ | ❌ | $0.075 / $0.30 |
| `liquid/lfm-2-24b-a2b` | 1 | Together (HIGH-RPM) | ❌ | ❌ | ❌ | ❌ | $0.03 / $0.12 |

แหล่งข้อมูล: `https://openrouter.ai/api/v1/models/<slug>/endpoints` (ดึง 2026-05-17)

## ผล Benchmark

| Model | Accuracy | p50 | p95 | Errors |
|---|---|---|---|---|
| `inception/mercury-2` | **100%** | 1028ms | 1398ms | 0 |
| `mistralai/devstral-small` | **100%** | **749ms** | 1476ms | 0 |
| `nvidia/nemotron-nano-9b-v2` | **100%** | 2744ms | 5230ms | 0 |
| `openai/gpt-5-nano` | **100%** | 3773ms | 7278ms | 0 |
| `amazon/nova-lite-v1` *(ปัจจุบัน)* | 96% | 952ms | 1537ms | 0 |
| `ibm-granite/granite-4.1-8b` | 96% | **725ms** | 1139ms | 0 |
| `openai/gpt-oss-safeguard-20b` | 88% | 574ms | 1265ms | 0 |
| `liquid/lfm-2-24b-a2b` | 80% | 961ms | 1409ms | 0 |

### รายละเอียดเคสที่พลาด

- **nova-lite-v1**: bare `"100"` หลัง AI พูด `"บันทึก…เรียบร้อยแล้ว ✓"` → classify เป็น add (เคส session boundary แบบ edge)
- **granite-4.1-8b**: พลาดเคสเดียวกับ Nova
- **gpt-oss-safeguard-20b**: พลาด `"เติมน้ำมัน 800"`, `"เที่ยว"`, `"200 บาท"` — accuracy ผันผวน (รัน 1: 96% / รัน 2: 88%) → ไม่ stable
- **lfm-2-24b-a2b**: 5 cases over-classify analytics/คำถามเป็น add_transaction → **ไม่ปลอดภัย**

## ข้อสังเกตสำคัญ

### 1. ทำไม p50/p95 latency สำคัญกว่าราคา headline

Classifier ถูกเรียก **ทุก text turn** ก่อน user เห็น response อะไรเลย → p50 กระทบความรู้สึกตอบสนองของแชทโดยตรง, p95 คุม worst-case lag ที่ผู้ใช้เห็นจริง

ดังนั้นต่อให้ accuracy 100% ถ้า p50 > 2 วินาทีก็ใช้ไม่ได้:
- `gpt-5-nano` (p50=3773ms, p95=7278ms) — ช้าเกินทั้งที่เป็น HA 3 endpoints
- `nemotron-nano-9b-v2` (p50=2744ms) — เป็น reasoning model มี hidden thinking tokens

### 2. Endpoint count แยกอิสระจาก accuracy

โมเดลที่ accuracy 100% บน single endpoint = accuracy 0% เมื่อ provider ล่ม Benchmark วัดคุณภาพในวันปกติ แต่ **redundancy** ตัดสินว่าเกิดอะไรในวันที่ provider มีปัญหา

มีแค่ 2 candidate ที่มี ≥ 2 endpoints คนละ infrastructure:
- `nova-lite-v1` (Bedrock × 2 regions) — provider เดียวแต่ multi-region
- `gpt-5-nano` (Azure × 2 + OpenAI-direct) — cross-infra HA ดีสุด แต่ช้าเกิน

ที่เหลือ (Mercury, Devstral, Granite, Safeguard, LFM, Nemotron) ทั้งหมดเป็น **SPOF** ถ้าไม่ตั้ง cross-model fallback chain

### 3. ความ stable ระหว่างรัน

- `gpt-oss-safeguard-20b` accuracy แกว่ง 96% ↔ 88% บนเคสชุดเดียวกัน (temperature=0) → เป็น non-determinism จาก Groq/model → swing ≥ 4% ต่อรันคือไม่เหมาะเป็น primary
- `nova-lite-v1`, `mercury-2`, `devstral-small`, `granite-4.1-8b` รันซ้ำได้ผลใกล้เคียงเดิม

## คำแนะนำ

3 ทางเลือก ตาม priority ต่างกัน:

### A. เน้น cost + speed (แนะนำถ้าเน้นถูก+เร็ว)
```
primary:  ibm-granite/granite-4.1-8b   (WandB)
fallback: amazon/nova-lite-v1           (Bedrock — คนละ infra)
```
- p50 **725ms** (เร็วที่สุดในกลุ่ม viable), accuracy 96% (พลาดเคสเดียวกับ Nova ปัจจุบัน)
- $0.05 / $0.10 ต่อ M tokens — **ถูกกว่า Nova ปัจจุบัน**
- Single endpoint ของ WandB แก้ด้วย Bedrock fallback

### B. เน้น quality (แนะนำถ้าเน้น accuracy)
```
primary:  mistralai/devstral-small     (Mistral)
fallback: amazon/nova-lite-v1          (Bedrock)
```
- 100% บนชุดทดสอบนี้, p50 **749ms** (ยังเร็ว)
- ราคา ~1.6× Nova ฝั่ง input, ~1.25× ฝั่ง output — เพิ่มน้อยเพราะ prompt สั้น
- แก้ปัญหาเคส bare-`"100"`-after-confirmation ที่ Nova พลาดได้

### C. Status quo (แนะนำถ้าเน้น HA สูงสุด, ไม่ต้องเปลี่ยน)
- ใช้ `amazon/nova-lite-v1` ต่อ — เป็นเจ้าเดียวที่มี **2 endpoints** ใน band latency ที่ยอมรับได้
- เคสที่พลาด 1 เคสเป็น edge case หายาก quick-add lane ก็ recover ได้โดย re-prompt

### Candidate ที่ถูกตัดออก
- `gpt-5-nano` — p50 3.7s ทำลาย hot-path UX แม้ HA ดีที่สุด
- `nemotron-nano-9b-v2` — p50 2.7s + ไม่มี `structured_outputs`
- `mercury-2` — accuracy 100% แต่ **แพงกว่า Devstral 3 เท่า** สำหรับผลลัพธ์เดียวกันและช้ากว่า
- `gpt-oss-safeguard-20b` — accuracy variance + Groq เป็น LOW-RPM pool ไม่ปลอดภัยสำหรับ hot path
- `lfm-2-24b-a2b` — accuracy 80%, ชอบ over-classify คำถามเป็น add

## ข้อควรระวังก่อน ship

1. **25 cases เป็น smoke test ไม่ใช่ validation** — ก่อน commit จริงควรรันบน dataset 200+ cases ที่สะท้อน distribution ของ traffic จริง
2. **Latency จาก client เดียว** — วัดจากตำแหน่งเดียว users บนมือถือเครือข่ายจริงอาจเห็น p95 ต่างจากนี้
3. **ไม่ได้วัด cold start** — p50 ทั้งหมดได้ประโยชน์จาก provider warm cache ในการรันถี่ ๆ; request แรกของชั่วโมงจะช้ากว่า
4. **ยังไม่ model cache savings** — system prompt ขนาดหลาย KB และ static ทั้งหมด ราคาตระกูล Nova/Anthropic จะลดเยอะเมื่อ `cache_control` ทำงาน Devstral ก็มี `input_cache_read` pricing → ต้อง re-bench cost projection หลังเห็น production traffic
5. **Context length** — node นี้ส่งแค่ไม่กี่ร้อย tokens → ความต่าง context_length (32K vs 400K) ไม่มีผลที่นี่

## วิธี Reproduce

```bash
cd mint_agentic
uv run python test_intent_bench_multi.py
```

แก้รายการ `MODELS` ใน script ได้ตามต้องการ รันครั้งหนึ่งใช้เวลา ~3 นาที สำหรับ 8 โมเดล × 25 cases
