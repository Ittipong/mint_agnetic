"""Curated financial knowledge tool — static strategies and advice."""

from langchain_core.tools import tool

_KNOWLEDGE: dict[str, str] = {
    "debt_snowball": """
กลยุทธ์ Debt Snowball (หิมะก้อนกลิ้ง)
• จ่ายขั้นต่ำทุกหนี้ก่อน จากนั้นเอาเงินที่เหลือจ่ายหนี้ที่ยอดคงเหลือน้อยที่สุดก่อน
• เมื่อหนี้ก้อนเล็กหมด นำเงินที่เคยจ่ายมาเพิ่มไปที่หนี้ก้อนถัดไป (snowball effect)
• ข้อดี: เห็นผลเร็ว → กระตุ้น momentum ทางจิตใจ เหมาะกับคนที่ต้องการแรงบันดาลใจ
• ข้อเสีย: จ่ายดอกเบี้ยรวมมากกว่า Avalanche
• เหมาะกับ: มีหนี้หลายก้อน และต้องการ quick win เพื่อสร้างวินัย
""",

    "debt_avalanche": """
กลยุทธ์ Debt Avalanche (หิมะถล่ม)
• จ่ายขั้นต่ำทุกหนี้ก่อน จากนั้นเอาเงินที่เหลือจ่ายหนี้ที่ดอกเบี้ยสูงที่สุดก่อน
• เมื่อหนี้ดอกสูงหมด นำเงินไปต่อที่หนี้ดอกถัดไป
• ข้อดี: ประหยัดดอกเบี้ยรวมมากที่สุด — วิธีที่คุ้มค่าที่สุดในเชิงคณิตศาสตร์
• ข้อเสีย: อาจใช้เวลานานกว่าจะเห็นหนี้ก้อนแรกหมด
• เหมาะกับ: คนที่มีวินัยสูง ต้องการประหยัดเงินสูงสุดในระยะยาว
""",

    "credit_card_trap": """
罗 กับดักบัตรเครดิต — วิธีหลุดและป้องกัน
1. จ่ายขั้นต่ำ = กับดัน: ดอกเบี้ย 18-28%/ปี ทบต้นทุกเดือน ยอดไม่ลดลงจริง
2. วิธีออก:
   • หยุดรูดบัตรที่มีหนี้ค้างทันที
   • จ่ายมากกว่าขั้นต่ำให้มากที่สุดทุกเดือน
   • พิจารณาขอ balance transfer ไปบัตรที่ดอก 0% ช่วง promo
   • ใช้ Avalanche: จ่ายบัตรดอกสูงก่อน
3. ป้องกันในอนาคต:
   • ตั้งจ่ายเต็มจำนวน (Full Statement) อัตโนมัติ
   • ใช้บัตรเดบิตสำหรับค่าใช้จ่ายประจำวัน
   • วางงบก่อนรูดเสมอ
""",

    "50_30_20_rule": """
กฎ 50/30/20 — กรอบงบประมาณขั้นพื้นฐาน
• 50% ความต้องการ (Needs): ค่าเช่า ค่าอาหาร ค่าน้ำไฟ ค่าเดินทาง
• 30% ความต้องการเพิ่มเติม (Wants): ท่องเที่ยว บันเทิง ร้านอาหาร
• 20% ออมและลงทุน (Savings): กองทุนฉุกเฉิน หุ้น กองทุน เงินเกษียณ

เริ่มต้น: คำนวณรายได้สุทธิ/เดือน คูณแต่ละ % ได้งบแต่ละหมวด
ปรับตาม: ถ้ามีหนี้ ให้ย้ายบางส่วนจาก 30% ไปชำระหนี้ก่อน
""",

    "emergency_fund": """
กองทุนฉุกเฉิน (Emergency Fund)
• เป้าหมาย: 3-6 เดือนของค่าใช้จ่ายรายเดือน
• เก็บไว้ที่ไหน: บัญชีออมทรัพย์ดอกสูง (High-yield savings) หรือตลาดเงิน — ต้องถอนได้ทันที
• ทำไมต้องมี: ป้องกันก่อหนี้เมื่อเกิดเหตุฉุกเฉิน (ตกงาน ค่ารักษาพยาบาล รถเสีย)

วิธีสร้าง:
1. เริ่มจากเป้า mini: 10,000 บาท → 1 เดือน → 3 เดือน
2. ตั้ง auto-transfer วันที่รับเงินเดือน ทำก่อนใช้จ่ายอื่น
3. อย่าแตะยกเว้นฉุกเฉินจริง — เติมกลับทันทีที่ใช้ไป
""",

    "compound_interest": """
ดอกเบี้ยทบต้น — ทั้งเพื่อนและศัตรู
• เพื่อน (เมื่อออม/ลงทุน): เงินต้น + ดอก สร้างดอกซ้อนดอก เวลา = อาวุธสำคัญ
  ตัวอย่าง: ลงทุน 5,000/เดือน ผลตอบแทน 8%/ปี → 30 ปี = ~7.4 ล้านบาท
• ศัตรู (เมื่อเป็นหนี้): บัตรเครดิต 2%/เดือน = 26.8%/ปีแบบทบต้น
  จ่ายขั้นต่ำ 1,000 บาทต่อเดือนจากยอดหนี้ 30,000 → ใช้เวลา 4+ ปี

Rule of 72: หาร 72 ด้วยอัตราดอกเบี้ย = ปีที่เงินจะเพิ่มเป็น 2 เท่า
ตัวอย่าง: 72 ÷ 8% = 9 ปี เงินจะ double
""",

    "budget_basics": """
หลักการตั้งงบประมาณ
1. Zero-based budgeting: ทุกบาทมีหน้าที่ รายได้ - ค่าใช้จ่าย - ออม = 0
2. Pay yourself first: โอนเงินออมก่อนทันทีที่รับเงินเดือน ไม่ใช่ออมสิ่งที่เหลือ
3. Track everything: บันทึกทุกรายจ่าย แม้แต่ซื้อน้ำ → เห็น pattern ที่ซ่อนอยู่
4. Review รายเดือน: ปรับงบตามความเป็นจริง ไม่ใช่ทำครั้งเดียวแล้วทิ้ง
5. ใช้แอป: แอปบันทึกรายรับรายจ่ายช่วยให้ระบบอัตโนมัติและเห็นภาพชัดขึ้น
""",

    "saving_strategies": """
กลยุทธ์การออมเงิน
1. Automatic savings: ตั้ง auto-transfer วันที่รับเงินเดือน — ลดการพึ่งพา willpower
2. Round-up savings: ปัดเศษทุกรายจ่ายขึ้น เก็บส่วนต่างออม
3. No-spend challenge: เลือก 1 สัปดาห์/เดือน ไม่จ่ายค่าใช้จ่ายที่ไม่จำเป็น
4. 52-week challenge: ออมเพิ่ม 100 บาท/สัปดาห์ (สัปดาห์ 1 = 100, สัปดาห์ 52 = 5,200)
   รวม 1 ปี = 137,800 บาท
5. Windfall rule: เงินพิเศษ (โบนัส/ภาษีคืน) → 50% ออม/ชำระหนี้, 50% ใช้จ่ายได้

วางเป้าหมายให้ SMART: Specific, Measurable, Achievable, Relevant, Time-bound
""",
}

_TOPIC_ALIASES: dict[str, str] = {
    "snowball": "debt_snowball",
    "avalanche": "debt_avalanche",
    "credit_card": "credit_card_trap",
    "creditcard": "credit_card_trap",
    "บัตรเครดิต": "credit_card_trap",
    "หนี้บัตรเครดิต": "credit_card_trap",
    "50/30/20": "50_30_20_rule",
    "503020": "50_30_20_rule",
    "emergency": "emergency_fund",
    "กองทุนฉุกเฉิน": "emergency_fund",
    "compound": "compound_interest",
    "ดอกเบี้ยทบต้น": "compound_interest",
    "budget": "budget_basics",
    "งบประมาณ": "budget_basics",
    "saving": "saving_strategies",
    "ออม": "saving_strategies",
    "การออม": "saving_strategies",
}


@tool
def get_financial_advice(topic: str) -> str:
    """Get curated financial strategies and advice on personal finance topics.

    Use this for general financial knowledge and strategies — NOT for the user's actual data.

    Available topics (use English key or Thai keyword):
    - debt_snowball / snowball: Pay smallest debts first
    - debt_avalanche / avalanche: Pay highest-interest debts first
    - credit_card_trap / บัตรเครดิต: How to escape credit card debt
    - 50_30_20_rule / budget: Budgeting framework
    - emergency_fund / กองทุนฉุกเฉิน: Building a safety net
    - compound_interest / ดอกเบี้ยทบต้น: Power of compound interest
    - saving_strategies / การออม: Practical saving techniques

    Args:
        topic: The financial topic to get advice on (English key or Thai keyword)

    Returns:
        Detailed strategy/advice text in Thai
    """
    key = _TOPIC_ALIASES.get(topic.lower(), topic.lower().replace(" ", "_"))
    content = _KNOWLEDGE.get(key)

    if content:
        return content.strip()

    available = ", ".join(_KNOWLEDGE.keys())
    return (
        f"ไม่พบข้อมูลสำหรับ '{topic}'\n"
        f"หัวข้อที่มี: {available}\n"
        f"ลองใช้ keyword ที่ตรงกว่านี้ครับ"
    )
