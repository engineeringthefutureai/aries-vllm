#!/bin/bash
# llama-benchy runs matching the llama.cpp baselines (results land in results/).
# Usage: scripts/bench.sh <label>        e.g. scripts/bench.sh chunked-mtp2
# Needs llama-benchy (pip install llama-benchy) on PATH, or BENCHY=/path/to/llama-benchy.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
LABEL="${1:?label}"; TS=$(date +%Y%m%d-%H%M); mkdir -p results
BENCHY="${BENCHY:-llama-benchy}"
COMMON=(--base-url "http://127.0.0.1:${PORT:-8080}/v1" --model "${SERVED_MODEL_NAME:-/models/$MODEL_DIR}"
        --tokenizer "${MODELS_DIR}/${MODEL_DIR}" --api-key "$VLLM_API_KEY" --format md)
"$BENCHY" "${COMMON[@]}" --pp 2048 --tg 128 --depth 0 8192 26624 --runs 3 --no-cache \
  --save-result "results/$TS-$LABEL-cold.md"
"$BENCHY" "${COMMON[@]}" --pp 256 --tg 64 --depth 8192 26624 --runs 3 \
  --save-result "results/$TS-$LABEL-cached.md"
echo "Note: benchy's 'cached' rows have been full re-prefills on every model tested here."
echo "Use make smoke for the cached-return number."
