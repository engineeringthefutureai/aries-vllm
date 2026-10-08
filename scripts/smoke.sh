#!/bin/bash
# Correctness + speed smoke test against the running server (host side).
#   1. /health and /v1/models
#   2. warm decode: Fibonacci prompt x3 -> completion_tokens / seconds (use runs 2-3)
#   3. cold prefill: ~15K-token argparse prompt with a random prefix, max_tokens 1
#   4. cached return: same prompt again -> should be a few seconds, not tens
# Reads VLLM_API_KEY, PORT and SERVED_MODEL_NAME from .env + the model preset.
# The chat_template_kwargs below turn thinking off on Qwen-style templates; other templates ignore them.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
URL="http://${HOST_ADDR:-127.0.0.1}:${PORT:-8080}"
H=(-H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json')
MODEL="${SERVED_MODEL_NAME:-/models/$MODEL_DIR}"

curl -sf "$URL/health" >/dev/null && echo "health: ok"
code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/models")
[[ "$code" == 401 ]] && echo "auth: ok (401 without key)" || { echo "auth: EXPECTED 401 WITHOUT KEY, GOT $code" >&2; exit 1; }
curl -sf "$URL/v1/models" "${H[@]}" | python3 -c 'import sys,json; print("models:", [m["id"] for m in json.load(sys.stdin)["data"]])'

echo "--- warm decode (Fibonacci, 300 tokens, thinking off)"
for i in 1 2 3; do
  s=$(date +%s.%N)
  n=$(curl -sf "$URL/v1/chat/completions" "${H[@]}" -d "{\"model\":\"$MODEL\",\"max_tokens\":300,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that returns the n-th Fibonacci number, with a docstring.\"}]}" \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["usage"]["completion_tokens"])')
  e=$(date +%s.%N)
  python3 -c "print(f'run $i: $n tokens in {$e-$s:.2f}s = {$n/($e-$s):.1f} tok/s')"
done

CTX=$(mktemp); REQ=$(mktemp); trap 'rm -f $CTX $REQ' EXIT
head -c 70000 "$(python3 -c 'import argparse; print(argparse.__file__)')" > "$CTX"
python3 - "$CTX" "$REQ" "$MODEL" <<'PY'
import json, sys, uuid
ctx, req, model = sys.argv[1:]
t = open(ctx).read()
json.dump({"model": model, "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False},
           "messages": [{"role": "user", "content": f"[run {uuid.uuid4()}]\n" + t + "\n\nSummarize this code in 5 bullets."}]},
          open(req, "w"))
PY
echo "--- cold prefill (~15K tokens, unique prefix, max_tokens 1)"
for label in cold cached; do
  s=$(date +%s.%N)
  p=$(curl -sf "$URL/v1/chat/completions" "${H[@]}" -d @"$REQ" | python3 -c 'import sys,json; print(json.load(sys.stdin)["usage"]["prompt_tokens"])')
  e=$(date +%s.%N)
  python3 -c "print(f'$label: $p prompt tokens in {$e-$s:.2f}s' + ('' if '$label'=='cached' else f' = {$p/($e-$s):.0f} tok/s'))"
done
echo "(first run after a restart includes kernel JIT; rerun before recording numbers)"
