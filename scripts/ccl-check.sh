#!/bin/bash
# Run the two-card all-reduce check in the image. vLLM must be stopped (GPUs in use).
# Usage: scripts/ccl-check.sh [chunked|fallback|raw]
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-chunked}"
case "$MODE" in
  chunked)  ENVS=(-e CCL_ENABLE_SYCL_KERNELS=1 -e PYTHONPATH=/opt/aries-vllm/pypatch) ;;
  fallback) ENVS=(-e CCL_ENABLE_SYCL_KERNELS=0) ;;
  raw)      ENVS=(-e CCL_ENABLE_SYCL_KERNELS=1) ;;
  *) echo "mode: chunked|fallback|raw" >&2; exit 1 ;;
esac
if docker ps --format '{{.Names}}' | grep -qx aries-vllm; then
  echo "aries-vllm is running; stop it first (make down) so the test gets the GPUs" >&2; exit 1
fi
docker compose run --rm --no-deps "${ENVS[@]}" -e VLLM_API_KEY=unused \
  --entrypoint python3 vllm /opt/aries-vllm/tests/ccl/ccl_check.py 2>&1 \
  | grep -E "^(CCL_|ok|BAD|FAIL)|chunked (all_reduce|XpuCommunicator)"
