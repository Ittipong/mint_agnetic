#!/usr/bin/env bash
# One NEW USER PLAYBOOK round: fresh user per group, writers run alone.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
export QA_USER=9c71ef82-9ddc-5624-a34f-29443b7f924e QA_WALLET=222b72e9-0b83-5450-91ae-b3237067b466
OUT=${1:-results}
rm -rf "$OUT" results
$PY seed_newuser.py 5000 >/dev/null
READONLY=$(python3 -c "import json;print(' '.join(s['id'] for s in json.load(open('scenarios_newuser.json')) if s['id'] not in ('N08_first_add','N09_first_income','N13_starter_flow')))")
set +e
for chunk in $(python3 -c "ids='$READONLY'.split(); print(' '.join(','.join(ids[i::4]) for i in range(4)))"); do
  python3 chat_qa.py scenarios_newuser.json ${chunk//,/ } >/dev/null 2>&1 &
done
wait
for id in N08_first_add N09_first_income N13_starter_flow; do
  $PY seed_newuser.py 5000 >/dev/null
  python3 chat_qa.py scenarios_newuser.json $id >/dev/null 2>&1
done
set -e
[ "$OUT" != results ] && mv results "$OUT"
$PY seed_newuser.py 5000 >/dev/null   # leave the user fresh
