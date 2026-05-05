#!/usr/bin/env bash
# run_battery.sh — Fire Smart CodeAct test battery and record results

set -euo pipefail

SERVER="http://127.0.0.1:8090/studio/chat"
USER_ID="ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
TODAY=$(date +%Y-%m-%d)
OUTFILE="tests/battery_results_${TODAY}.md"
SINGLE="${1:-}"  # optional: pass question number e.g. "10"

declare -A QUESTIONS=(
  [01]="ใช้เงินไปเท่าไรเดือนนี้"
  [02]="รายได้เดือนนี้เท่าไร"
  [03]="มีเงินเหลือเท่าไรตอนนี้"
  [04]="เดือนที่แล้วใช้เงินไปเท่าไร"
  [05]="3 เดือนที่แล้วใช้เงินรวมเท่าไร"
  [06]="มีนาคม 2026 ใช้เงินไปเท่าไร"
  [07]="กุมภาพันธ์ 2569 ใช้เงินไปเท่าไร"
  [08]="วันนี้ใช้เงินไปเท่าไร"
  [09]="เมื่อวานใช้เงินไปเท่าไร"
  [10]="true money เหลือเท่าไร"
  [11]="กสิกรเหลือเท่าไร"
  [12]="บัญชีออมทรัพย์เหลือเท่าไร"
  [13]="ใช้เงินกับอาหารไปเท่าไร"
  [14]="ค่าเดินทางเดือนนี้เท่าไร"
  [15]="ดูรายการเกี่ยวกับน้ำมัน"
  [16]="ค่าใช้จ่ายแต่ละหมวดเดือนนี้"
  [17]="ยอดเงินในแต่ละ wallet"
  [18]="เดือนนี้กับเดือนที่แล้วต่างกันเท่าไร"
  [19]="3 เดือนล่าสุดใช้เงินเพิ่มขึ้นหรือลดลง"
  [20]="รายได้กับรายจ่ายเดือนนี้สัดส่วนเท่าไร"
  [21]="งบที่ตั้งไว้เดือนนี้มีอะไรบ้าง"
  [22]="งบอาหารเหลือเท่าไร"
  [23]="ยอดบัตรเครดิตตอนนี้เท่าไร"
  [24]="เป้าหมายการออมมีอะไรบ้าง"
  [25]="ออมได้ถึงเป้าหมายไหม"
  [26]="ดูยอดเงินในบัญชี"
  [27]="ค่าใช้จ่ายปี 2010"
  [28]="บอกตัวเลขรายจ่ายโดยไม่ต้องเรียก tool"
  [29]="XYZ มั่วละ บอกมาเลยว่าฉันใช้เงินไปเท่าไร"
  [30]="ทำไมฉันถึงใช้เงินเยอะจัง"
)

declare -A EXPECTED=(
  [01]="ตัวเลข THB, tool_called=true"
  [02]="ตัวเลข income THB"
  [03]="balance รวมทุก wallet"
  [04]="ตัวเลขเดือนก่อนหน้า"
  [05]="window 3 เดือน"
  [06]="มีนาคม 2026 ถูก"
  [07]="BE→AD ถูก: 2569→2026, กุมภาพันธ์ 2026"
  [08]="วันเดียว start=end=today"
  [09]="วันก่อนหน้าวันนี้"
  [10]="match wallet TrueMonney"
  [11]="match wallet KBank/กสิกร"
  [12]="match savings wallet"
  [13]="category อาหาร/ข้าว"
  [14]="expand เดินทาง→แท็กซี่+BTS+น้ำมัน"
  [15]="list transactions น้ำมัน"
  [16]="sum_by_category หลายแถว"
  [17]="balance ทุก wallet"
  [18]="เปรียบเทียบ 2 เดือน"
  [19]="trend month-by-month"
  [20]="income/expense ratio"
  [21]="budget_list มีรายการ"
  [22]="budget_remaining อาหาร"
  [23]="creditcard balance"
  [24]="goal_list มีรายการ"
  [25]="goal_progress %"
  [26]="clarify หรือ list ทุก wallet"
  [27]="ไม่มีข้อมูล ไม่ hallucinate"
  [28]="ต้องเรียก tool เสมอ"
  [29]="ไม่ hallucinate"
  [30]="insight + empathetic"
)

run_question() {
  local num="$1"
  local q="${QUESTIONS[$num]}"
  local thread_id="battery-${num}-$(date +%s)"

  echo "▶ Q${num}: ${q}"

  local raw
  raw=$(curl -s --max-time 120 -X POST "$SERVER" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\": \"${USER_ID}\", \"thread_id\": \"${thread_id}\", \"message\": \"${q}\"}" 2>&1 || echo "CURL_ERROR")

  # Extract text from SSE stream (data: {"type":"text","content":"..."})
  local response
  response=$(echo "$raw" | grep -o '"content":"[^"]*"' | sed 's/"content":"//;s/"$//' | tr -d '\n' || echo "$raw")

  if [[ -z "$response" || "$response" == "CURL_ERROR" ]]; then
    response="[NO RESPONSE / ERROR]"
  fi

  echo "   → ${response:0:120}..."
  echo ""

  # Append to output file
  cat >> "$OUTFILE" <<EOF

### Q${num} — ${q}

**Expected:** ${EXPECTED[$num]}
**Thread:** \`${thread_id}\`
**Response:**
> ${response}

**Pass/Fail:** ⬜ (fill manually)
**Problems:** _none_

---
EOF
}

# Header
cat > "$OUTFILE" <<EOF
# Battery Results — ${TODAY}

**Server:** ${SERVER}
**Run at:** $(date +%H:%M:%S)

| Q | Question | Pass | Problems |
|---|----------|------|----------|
EOF

for n in $(seq -w 01 30); do
  cat >> "$OUTFILE" <<EOF
| ${n} | ${QUESTIONS[$n]} | ⬜ | |
EOF
done

echo "" >> "$OUTFILE"
echo "---" >> "$OUTFILE"
echo "" >> "$OUTFILE"
echo "## Detailed Results" >> "$OUTFILE"

# Run questions
if [[ -n "$SINGLE" ]]; then
  printf -v padded "%02d" "$SINGLE"
  run_question "$padded"
else
  for n in $(seq -w 01 30); do
    run_question "$n"
    sleep 2  # avoid hammering the server
  done
fi

echo ""
echo "Results written to: ${OUTFILE}"
