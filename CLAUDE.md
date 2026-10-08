# Aries vLLM: tensor parallelism on 2x Intel Arc Pro B60

Read README.md first: it explains every setting and why it's there. Several settings that look
redundant are load-bearing, and most "obvious" fixes (other oneCCL knobs, graph capture, fp8 KV)
were tried and failed.

## Layout

- `docker/entrypoint.sh`: env → `vllm serve` flags. Hardware/stack defaults only (TP=2, eager,
  all-reduce mode); model-specific values belong in `models/*.env`, never as entrypoint defaults.
- `docker/Dockerfile`: fixed env (oneCCL, XPU) with a comment per setting.
- `docker/pypatch/sitecustomize.py`: chunked all-reduce patch (see README, "All-reduce").
- `models/<name>.env`: one preset per model, each value commented with why.
- `tests/ccl/ccl_check.py`: two-GPU all-reduce check, run via `make ccl-check`.
- `scripts/`: `smoke.sh`, `bench.sh`, `ccl-check.sh`; `env.sh` loads preset + `.env` literally.
- `docs/results.md`: reference numbers per config and version.
- `VERSION`: our semver; the base image is pinned (tag + digest) in the Makefile. Release
  checklist and tag scheme: README, "Versioning".

## Rules

- **Measure, don't estimate.** Performance or stability claims need `make smoke` / `make bench`
  numbers; label anything else as an estimate. New configs get a row in `docs/results.md`.
- **One change per experiment.** Don't bundle knob changes, or the numbers can't be attributed.
- **Read the output, not just the speed.** Fast garbage is a known failure mode (README,
  "Troubleshooting").
- **Never print the API key.** It comes from `.env` as `VLLM_API_KEY`; vLLM reads it from the
  environment. Don't `cat .env`, don't pass `--api-key` to vLLM, and don't run plain
  `docker compose config` (it expands env_file values; `make config` doesn't).
- The server binds 127.0.0.1 by default; the entrypoint refuses to start without a key and warns
  on any other HOST. vLLM's key only guards /v1/*, so network exposure goes through a reverse
  proxy (README, "Network exposure"). Keep both defaults.
- `make ccl-check` needs both GPUs free (`make down` first). After changing the all-reduce patch,
  both `dist` and `vllm` rows must pass; the `vllm` path is what serving uses.
- Don't install Intel's offline installer or change the host driver stack; the container brings
  its own userspace.
- Only one inference server can own both GPUs and the port at a time.
