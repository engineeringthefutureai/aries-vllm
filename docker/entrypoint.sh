#!/bin/bash
# Builds the `vllm serve` command from environment variables.
#
# Defaults here are hardware/stack facts for 2x Arc Pro B60 (TP=2, eager mode, all-reduce
# mode). Everything model-specific comes from a preset (MODEL_PRESET -> models/<name>.env);
# variables already set in the environment win over the preset. Unset optional knobs are
# not passed, so vLLM's own defaults apply. Extra CLI args given to the container are
# appended verbatim, after VLLM_EXTRA_ARGS.
set -euo pipefail

if [[ -n "${MODEL_PRESET:-}" ]]; then
  preset="/opt/aries-vllm/models/${MODEL_PRESET}.env"
  if [[ ! -f "$preset" ]]; then
    echo "[aries-vllm] no preset $preset; available:" $(ls /opt/aries-vllm/models | sed 's/\.env$//') >&2
    exit 1
  fi
  # Literal KEY=value lines, like docker --env-file (JSON values keep their quotes).
  while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" == \#* ]] && continue
    if [[ -z "${!k+x}" ]]; then export "$k=$v"; fi
  done < "$preset"
fi

: "${HOST:=127.0.0.1}"            # loopback only; see README, "Network exposure"
: "${PORT:=8080}"
: "${TP_SIZE:=2}"                 # both cards compute every token
: "${ENFORCE_EAGER:=1}"           # graph capture is untested here and breaks hybrid (GDN) models
: "${ARIES_VLLM_CCL_MODE:=chunked}"    # chunked | fallback   (see README, "All-reduce")

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "[aries-vllm] VLLM_API_KEY is not set; refusing to start an unauthenticated server" >&2
  exit 1
fi
# vLLM reads VLLM_API_KEY from the environment, so the key never appears in argv or the log.
case "$HOST" in
  127.0.0.1|localhost|::1) ;;
  *) echo "[aries-vllm] WARNING: listening on $HOST. vLLM's API key only guards /v1/*; other routes" \
          "(/invocations, /tokenize, /pause, ...) answer without it. See README, \"Network exposure\"" \
          "and https://docs.vllm.ai/en/latest/usage/security/" >&2 ;;
esac

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "[aries-vllm] set MODEL_PRESET (see models/), or MODEL_DIR (checkpoint dir under /models)" >&2
    exit 1
  fi
  MODEL_PATH="/models/$MODEL_DIR"
fi

case "$ARIES_VLLM_CCL_MODE" in
  chunked)
    export CCL_ENABLE_SYCL_KERNELS=1
    export PYTHONPATH="/opt/aries-vllm/pypatch${PYTHONPATH:+:$PYTHONPATH}" ;;
  fallback)
    export CCL_ENABLE_SYCL_KERNELS=0 ;;
  *) echo "[aries-vllm] unknown ARIES_VLLM_CCL_MODE=$ARIES_VLLM_CCL_MODE (chunked|fallback)" >&2; exit 1 ;;
esac

args=(
  "$MODEL_PATH"
  --host "$HOST" --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --disable-custom-all-reduce     # no XPU custom all-reduce in this stack; oneCCL does it
)
[[ "$ENFORCE_EAGER" == 1 ]]         && args+=(--enforce-eager)
[[ -n "${SERVED_MODEL_NAME:-}" ]]   && args+=(--served-model-name "$SERVED_MODEL_NAME")
[[ -n "${MAX_MODEL_LEN:-}" ]]       && args+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "${GPU_MEMORY_UTIL:-}" ]]     && args+=(--gpu-memory-utilization "$GPU_MEMORY_UTIL")
[[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] && args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
[[ -n "${KV_CACHE_DTYPE:-}" ]]      && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${LIMIT_MM_PER_PROMPT:-}" ]] && args+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
[[ -n "${SPECULATIVE_CONFIG:-}" ]]  && args+=(--speculative-config "$SPECULATIVE_CONFIG")
[[ -n "${REASONING_PARSER:-}" ]]    && args+=(--reasoning-parser "$REASONING_PARSER")
[[ -n "${TOOL_CALL_PARSER:-}" ]]    && args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
[[ -n "${GENERATION_CONFIG:-}" ]]   && args+=(--override-generation-config "$GENERATION_CONFIG")
case "${PREFIX_CACHING:-}" in
  1) args+=(--enable-prefix-caching) ;;
  0) args+=(--no-enable-prefix-caching) ;;
esac
# Anything else: space-separated flags, no quoting (put JSON values in the knobs above).
# shellcheck disable=SC2206
[[ -n "${VLLM_EXTRA_ARGS:-}" ]] && args+=($VLLM_EXTRA_ARGS)

echo "[aries-vllm] ${ARIES_VLLM_VERSION:-dev} preset=${MODEL_PRESET:-none} model=$MODEL_PATH tp=$TP_SIZE ccl=$ARIES_VLLM_CCL_MODE eager=$ENFORCE_EAGER" \
     "ctx=${MAX_MODEL_LEN:-default} kv=${KV_CACHE_DTYPE:-default} spec=${SPECULATIVE_CONFIG:-off}" >&2
exec vllm serve "${args[@]}" "$@"
