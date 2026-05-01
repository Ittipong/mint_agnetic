#!/bin/bash
# LangGraph Studio Chat API Test Script
# Usage: ./test_studio_chat.sh [message]

HOST="${HOST:-http://localhost:8000}"
USER_ID="${USER_ID:-ba91d8a5-46b2-46f7-aaf4-189a54e17fe9}"
THREAD_ID="${THREAD_ID:-test-$(date +%s)}"
MESSAGE="${1:-ใช้เงินไปเท่าไร}"

echo "=== LangGraph Studio Chat Test ==="
echo "Host: $HOST"
echo "User ID: $USER_ID"
echo "Thread ID: $THREAD_ID"
echo "Message: $MESSAGE"
echo ""

# Stream response
echo "--- Streaming Response ---"
curl -s -N -X POST "$HOST/studio/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"user_id\": \"$USER_ID\",
    \"thread_id\": \"$THREAD_ID\",
    \"message\": \"$MESSAGE\"
  }"

echo ""
echo ""
echo "=== Done ==="
