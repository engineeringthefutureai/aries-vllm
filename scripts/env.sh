# Sourced by the scripts: loads models/$MODEL_PRESET.env, then .env (later wins), reading
# values literally like docker --env-file (no quote removal, so JSON values survive).
_aries_vllm_load() {
  local k v
  while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" == \#* ]] && continue
    export "$k=$v"
  done < "$1"
}
[[ -f .env ]] || { echo "no .env; cp .env.example .env" >&2; exit 1; }
_aries_vllm_load .env
if [[ -n "${MODEL_PRESET:-}" ]]; then
  [[ -f "models/$MODEL_PRESET.env" ]] || { echo "no preset models/$MODEL_PRESET.env" >&2; exit 1; }
  _aries_vllm_load "models/$MODEL_PRESET.env"
fi
_aries_vllm_load .env
