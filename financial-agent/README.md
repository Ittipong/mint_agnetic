# Financial Agent (ผู้ช่วยวิเคราะห์การเงินอัจฉริยะ)

Financial Agent คือผู้ช่วยทางการเงินที่ขับเคลื่อนด้วย AI ซึ่งสร้างขึ้นบนสถาปัตยกรรม **CodeAct** ระบบสามารถโต้ตอบกับฐานข้อมูลทางการเงิน (PostgreSQL) ของคุณด้วยคำสั่งภาษาธรรมชาติ (Natural Language) โดยทำงานร่วมกับ LLM (ผ่าน OpenRouter) เพื่อสร้างและรันโค้ด Python อย่างปลอดภัยในการวิเคราะห์รายการธุรกรรม กระเป๋าเงิน งบประมาณ และข้อมูลทางการเงินอื่นๆ

## คุณสมบัติเด่น (Features)

- **สั่งงานด้วยภาษาธรรมชาติ**: สอบถามข้อมูลทางการเงินของคุณด้วยภาษาพูดทั่วไป
- **ระบบ CodeAct Loop**: สามารถสร้างโค้ด, รันโค้ด, และปรับปรุงโค้ดซ้ำๆ เพื่อหาคำตอบที่ถูกต้องที่สุด
- **รองรับคำถามที่ซับซ้อน (Complex Queries)**: สามารถใช้โหมด `--complex` เพื่อให้ AI ใช้เวลาคิดวิเคราะห์เชิงลึกมากขึ้น
- **Sandbox Execution**: รันโค้ดได้อย่างปลอดภัยผ่าน `quantalogic-pythonbox`

## ตัวอย่างหน้าตาการใช้งาน (ASCII UI Preview)

```text
+-------------------------------------------------------------+
| Terminal                                            [-][x]  |
+-------------------------------------------------------------+
| $ make run TASK="สรุปค่าใช้จ่ายเดือนนี้ให้หน่อย"                  |
|                                                             |
| 🤖 Financial Agent:                                         |
| ┌─────────────────────────────────────────────────────────┐ |
| │ [Step 1] Analyzing query...                             │ |
| │ [Step 2] Generating Python code...                      │ |
| │ [Step 3] Executing in secure Sandbox...                 │ |
| │ [Step 4] Extracting results...                          │ |
| └─────────────────────────────────────────────────────────┘ |
|                                                             |
| ✅ ผลลัพธ์: ค่าใช้จ่ายเดือนนี้ของคุณคือ 15,400 บาท                |
|                                                             |
| 📊 รายละเอียด:                                                |
| - 🍔 อาหาร: 5,200 THB                                       |
| - 🚗 เดินทาง: 3,000 THB                                     |
| - 🛒 ช้อปปิ้ง: 7,200 THB                                      |
+-------------------------------------------------------------+
```

## สถาปัตยกรรม (Architecture)

สถาปัตยกรรมของ Financial Agent ประกอบด้วยส่วนประกอบหลักดังนี้:

- **Client / CLI**: จุดเชื่อมต่อที่รับคำสั่งภาษาธรรมชาติและ User ID จากผู้ใช้
- **CodeAct Loop**: ตัวกลางควบคุมการทำงานหลัก (Orchestrator) ที่คอยจัดการลูปของการคิดวิเคราะห์ (Reasoning) และการรันโค้ด (Executing)
- **Reasoner**: ส่วนที่เชื่อมต่อกับ LLM (เช่น Gemini) ทำหน้าที่รับคำสั่ง, บริบทโครงสร้างฐานข้อมูล (DB Schema), และประวัติการทำงาน เพื่อวิเคราะห์และสร้างโค้ด Python 
- **Executor**: ระบบรันโค้ดในพื้นที่ปลอดภัย (Sandbox) โดยใช้ `quantalogic-pythonbox`
- **Database**: ฐานข้อมูล PostgreSQL ที่จัดเก็บข้อมูลทางการเงินต่างๆ (Transactions, Wallets, Budgets ฯลฯ)

## ขั้นตอนการทำงาน (Workflow)

1. **รับคำสั่ง**: ผู้ใช้ป้อนคำสั่ง (Task) เช่น "สรุปรายจ่ายเดือนนี้" พร้อมกับ User ID
2. **เริ่ม CodeAct Loop**: ระบบเริ่มลูปการทำงาน ซึ่งสามารถวนซ้ำได้ตามจำนวนรอบสูงสุดที่กำหนด (Max Steps)
3. **วิเคราะห์และสร้างโค้ด (Reasoning)**: `Reasoner` ส่งข้อมูลประกอบด้วย คำสั่ง, Schema ของฐานข้อมูล, และประวัติผลลัพธ์จากรอบก่อนหน้าไปยัง LLM เพื่อให้ LLM สร้างโค้ด Python
4. **รันโค้ดอย่างปลอดภัย (Executing)**: โค้ด Python ที่ได้จะถูกนำไปรันใน `Executor` (Sandbox) โดยตัวโค้ดจะคิวรีข้อมูลจากฐานข้อมูล PostgreSQL
5. **ประเมินผลลัพธ์ (Evaluation)**: ผลลัพธ์จากการรันโค้ด (ไม่ว่าจะเป็นข้อมูลที่ดึงมาได้ หรือ Error) จะถูกส่งกลับไปที่ลูป
   - หากได้คำตอบที่สมบูรณ์ ลูปจะสิ้นสุด
   - หากข้อมูลยังไม่ครบถ้วน หรือมี Error ลูปจะวนกลับไปที่ข้อ 3 เพื่อให้ LLM แก้ไขโค้ดใหม่
6. **คืนค่าผลลัพธ์**: ส่งคำตอบสุดท้ายกลับไปยังผู้ใช้

## แผนภาพการทำงาน (Diagram)

> [ดูแผนภาพขั้นตอนการทำงาน (Workflow Diagram) แบบเต็มได้ที่นี่](./workflow-diagram.mermaid)

## สิ่งที่ต้องเตรียม (Prerequisites)

- Python 3.11+
- เครื่องมือจัดการแพ็กเกจ `uv` (แนะนำ)
- ฐานข้อมูล PostgreSQL

## การติดตั้ง (Installation)

สามารถติดตั้งโปรเจกต์และ dependencies ได้ผ่าน `uv` โดยใช้ `Makefile` ที่เตรียมไว้:

```bash
# ติดตั้งแพ็กเกจหลัก
make install

# ติดตั้งแพ็กเกจสำหรับการพัฒนา (รวมถึง pytest)
make install-dev
```

## การตั้งค่า (Configuration)

ระบบใช้ environment variables ในการตั้งค่า ให้คัดลอกไฟล์ `.env.example` มาสร้างเป็น `.env` ของคุณ:

```bash
cp .env.example .env
```

ตัวแปรในไฟล์ `.env` ที่สำคัญ:

- `OPENROUTER_API_KEY`: API Key สำหรับใช้งาน OpenRouter
- `DATABASE_URL`: Connection String ของ PostgreSQL (ค่าเริ่มต้น: `postgresql+asyncpg://postgres:postgres@localhost:5432/mint_money_dev`)
- `OPENROUTER_BASE_URL`: Base URL ของ OpenRouter
- `MODEL`: รุ่นของ LLM ที่ใช้งานปกติ (ค่าเริ่มต้น: `google/gemini-2.5-flash-lite`)
- `COMPLEX_MODEL`: รุ่นของ LLM สำหรับโหมดคิดวิเคราะห์ซับซ้อน (ค่าเริ่มต้น: `google/gemini-2.5-flash-lite`)
- `MAX_STEPS`: จำนวนรอบสูงสุดในการทำงานของ CodeAct loop (ค่าเริ่มต้น: `3`)
- `EXECUTOR_TIMEOUT`: เวลาสูงสุดในการรันโค้ด (ค่าเริ่มต้น: `30` วินาที)
- `SOLVE_TIMEOUT`: เวลาสูงสุดในการหาคำตอบทั้งหมด (ค่าเริ่มต้น: `180` วินาที)
- `LOG_LEVEL`: ระดับการเก็บ Log (ค่าเริ่มต้น: `INFO`)

## การใช้งาน (Usage)

สามารถใช้คำสั่งใน `Makefile` เพื่อทดสอบหรือรันระบบได้อย่างง่ายดาย

### ตัวอย่างทั่วไป

รันระบบด้วย User และคำสั่งเริ่มต้น:
```bash
make run
```

รันระบบพร้อมระบุคำสั่ง (Task):
```bash
make run TASK="show my wallets"
```

รันระบบพร้อมระบุ User ID และคำสั่ง:
```bash
make run USER_ID="<your-user-uuid>" TASK="How much did I spend on food last week?"
```

### โหมดซับซ้อน (Complex Mode)

สำหรับคำถามที่ต้องการการคิดวิเคราะห์เชิงลึก หรือเชื่อมโยงข้อมูลหลายส่วน ให้ใช้ `run-complex`:
```bash
make run-complex TASK="forecast my expenses for next month based on last 3 months"
```

### การใช้งานผ่าน CLI โดยตรง

คุณสามารถรันโปรแกรมผ่าน `uv` ได้โดยตรง:
```bash
uv run financial-agent --user-id "<uuid>" "What is my current balance?"
```

ตัวเลือกสำหรับ CLI:
- `--user-id`: (บังคับ) รหัสผู้ใช้ (UUID)
- `--complex`: เปิดใช้งานโหมดการวิเคราะห์แบบละเอียด (Extended thinking)
- `--dev`: เปิดโหมดการพัฒนา (แสดง Log ระดับ DEBUG และออกผลลัพธ์ทุกขั้นตอนในรูปแบบ JSON)

## การพัฒนาและแก้ไขปัญหา (Development & Debugging)

เพื่อดูรายละเอียดการทำงานของระบบในแต่ละสเต็ป สามารถเปิดโหมด Dev ได้:

```bash
# รันในโหมด Dev
make dev TASK="my expenses"

# รันโหมด Dev และบันทึกประวัติทุกสเต็ปลงในไฟล์ JSONL
make dev-log TASK="my expenses"
```

## การทดสอบระบบ (Testing & Benchmarks)

โปรเจกต์นี้ใช้ `pytest` ในการรันเทส:

```bash
# รัน Unit tests (ไม่จำเป็นต้องเชื่อมต่อ DB/API)
make test

# รัน End-to-end tests (ต้องเชื่อมต่อ DB และใช้ API key)
make test-e2e

# รันเทสทั้งหมด
make test-all

# รันระบบ Benchmark
make bench
```

## การตรวจสอบคุณภาพโค้ด (Code Quality)

คำสั่งสำหรับจัดรูปแบบและตรวจสอบโค้ด:

```bash
# ตรวจสอบสไตล์โค้ดด้วย Ruff
make lint

# จัดรูปแบบโค้ดอัตโนมัติ
make format

# ลบไฟล์แคชและไฟล์ที่เกิดจากการ Build
make clean
```
