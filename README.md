# Aries vLLM

A Docker image for serving LLMs with **vLLM tensor parallelism across two Intel Arc Pro B60
24 GB cards**: one OpenAI-compatible endpoint, both GPUs computing every token, 48 GB of VRAM.
Other Arc Pro B cards (B50, B65, B70) should work too; see [Hardware](#other-arc-pro-b-cards).

It is a thin layer over Intel's [LLM Scaler vLLM image](https://github.com/intel/llm-scaler/tree/main/vllm)
that bakes in what tensor parallelism needs on this hardware: oneCCL settings that don't crash,
a patch that makes the fast all-reduce path usable (1.7x prefill, 1.4x decode over the safe
fallback), eager mode for hybrid models, and an entrypoint driven by `.env` plus per-model presets.

**XPU** is the name vLLM and PyTorch use for Intel GPUs (the way `cuda` means NVIDIA GPUs): the
B60s run on vLLM's XPU backend, see
[vLLM's Intel XPU docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#intel-xpu).
Anything named XPU below (`VLLM_XPU_*` variables, `XpuCommunicator`, "XPU workers") refers to it.

## Models

| Preset | Model | Status | Context | Cold prefill | Decode |
|---|---|---|---|---|---|
| `qwen3.8-27b` | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (+ MTP) | validated | 98K | ~1,200 tok/s | 33–36 tok/s |
| `devstral-small-2-24b` | [mistralai/Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) (FP8) | experimental (smoke + bench only) | 131K | ~1,900 tok/s | 31 → 27 tok/s (0 → 26K deep) |
| `qwen3.5-9b` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (bf16) + MTP | experimental (smoke + bench only) | 262K | ~3,400 tok/s | 64–70 tok/s |
| `gemma-4-26b-a4b` | [RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic) | experimental (smoke + bench only) | 262K | 643 → 231 tok/s (0 → 26K deep) | 27 → 21 tok/s |

One validated model so far. Other models vLLM supports on XPU should work with a new preset (see
[Adding a model](#adding-a-model)), but they are untested here. Full numbers, including a
llama.cpp comparison on the same cards: [docs/results.md](docs/results.md).

## Hardware

Reference system, on which all numbers were measured:

| Component | Reference system |
|---|---|
| GPUs | 2x Intel Arc Pro B60 24 GB (Battlemage), host `xe` kernel driver |
| PCIe | card 1 Gen5 x8, **card 2 Gen4 x4**, both on CPU lanes (no chipset hop), no risers |
| CPU / RAM | Core Ultra 5 250K Plus, 32 GB (30 GiB usable) |
| OS | Ubuntu 26.04, Docker with the compose plugin |
| Kernel / firmware | Linux 7.0 (`xe` driver), GuC firmware 70.58.0 (linux-firmware) |

### Other Arc Pro B cards

Nothing here checks for a B60: the image, all-reduce patch and entrypoint are generic Battlemage
(Xe2) and the LLM Scaler base image targets the whole B family. Presets and numbers are what's
card-specific, mostly through VRAM. Only the B60 has been tested; reports for the rest are welcome.

| Setup | Chip, VRAM per card | Status | Notes |
|---|---|---|---|
| 2x B60 | BMG-G21, 24 GB | **validated** | reference system |
| Dual-GPU B60 board | 2x BMG-G21, 24 GB each | expected to work | needs a motherboard slot with x8/x8 bifurcation (BIOS setting). Then it's equivalent to two B60s on two x8 slots: a wider link for card 2 than the reference x4 |
| 2x B65 | BMG-G31, 32 GB | expected to work | more KV headroom: the Qwen preset's context can likely grow (estimate) |
| 2x B70 | BMG-G31, 32 GB | expected to work | as B65, plus more compute and memory bandwidth; 2x B70 runs this model in [llm-scaler #672](https://github.com/intel/llm-scaler/issues/672) |
| 2x B50 | BMG-G21, 16 GB | needs its own preset | 32 GB total: the 27B FP8 preset (~29 GB weights) doesn't fit; use a smaller or INT4 model |
| Mixed (e.g. B60 + B70) | — | untested | TP splits work evenly, so the smaller and slower card sets the pace |

If the >4 MB all-reduce failure (see [All-reduce](#all-reduce)) doesn't occur on your cards, the
chunked patch is harmless: decode-sized calls pass through unchanged.

An x4 second slot is what most consumer boards give a second GPU, and the defaults are tuned for
it. Here that x4 link comes from the CPU. On many boards it comes from the chipset instead,
sharing the chipset's uplink with NVMe, network and USB; that's untested and may be slower. An
x8/x8 board should do at least as well (more all-reduce bandwidth); also untested.

To see where your cards sit: `lspci -tv` shows each card's root port. On Arrow Lake / Z890, chipset
devices are in their own PCI domain (`0000:80`), so a card under a `0000:00` root port is on CPU
lanes. The negotiated link is on the upstream port of the card's onboard switch (the GPU itself
reports an internal 2.5 GT/s x1 link, which is normal):
`cat /sys/bus/pci/devices/<switch>/current_link_{speed,width}`.

The host only needs the `xe` kernel driver and GPU firmware; the container brings its own GPU
userspace (Level Zero, compute runtime, oneCCL, SYCL). Host Mesa (Vulkan/OpenGL) and host
compute-runtime packages aren't used by the container, so their versions don't matter here.

You don't need Intel's host installer from the LLM Scaler docs
(`multi-arc-bmg-offline-installer`): it replaces the kernel, GPU firmware and driver with Intel's
versions and targets a fresh Ubuntu 24.04. A distro kernel with the `xe` driver is enough, as on
the reference system.

## Quick start

```bash
git clone https://github.com/engineeringthefutureai/aries-vllm && cd aries-vllm
cp .env.example .env            # set VLLM_API_KEY and MODELS_DIR
huggingface-cli download Qwen/Qwen3.8-27B-FP8 --local-dir "$MODELS_DIR/qwen3.8-27b-fp8"

make pull                       # prebuilt image for this checkout (or: make build)
make ccl-check                  # all rows "ok": the cross-card all-reduce works on your box
make up                         # starts the server, waits for /health (a few minutes)
make smoke                      # auth, decode, cold prefill, cached return
```

The server listens on `127.0.0.1:8080` by default (loopback only) and refuses to start without
`VLLM_API_KEY`. Clients on the same host use `http://127.0.0.1:8080/v1` with the key as bearer
token. To serve other machines, read [Network exposure](#network-exposure) first. `make help`
lists all targets.

### Without the repo

The image is [`engineeringthefuture/aries-vllm`](https://hub.docker.com/r/engineeringthefuture/aries-vllm) and presets are baked
into it, so plain `docker run` with your `.env` works:

```bash
docker run -d --name aries-vllm \
  --privileged --network host --ipc host --shm-size 16g --memory 20g \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v /srv/models:/models:ro \
  --env-file .env \
  engineeringthefuture/aries-vllm:0.1.0-llmscaler0.26.0-b2
```

`--env-file` takes lines literally: no quotes, no trailing comments.

## Configuration

Settings come from `.env` (yours, gitignored) and a **model preset** (`models/<name>.env`) chosen
with `MODEL_PRESET`. Values in `.env` win over the preset, so you can override anything without
editing presets. Optional knobs that are unset aren't passed, so vLLM's defaults apply.

### Your settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_API_KEY` | — (required) | Bearer token. Read by vLLM from the environment, so it stays out of argv and logs |
| `MODELS_DIR` | — (required, compose) | Host directory with checkpoints, mounted at `/models` |
| `MODEL_PRESET` | — | Preset name, e.g. `qwen3.8-27b` |
| `HOST` | `127.0.0.1` | Listen address. `0.0.0.0` exposes the server to the network; read [Network exposure](#network-exposure) |
| `PORT` | `8080` | Listen port |
| `ARIES_VLLM_IMAGE` | `engineeringthefuture/aries-vllm` | Image to build or run |
| `ARIES_VLLM_TAG` | this checkout's version via `make`; `latest` via plain compose | Pin a release, e.g. `0.1.0-llmscaler0.26.0-b2` (see [Versioning](#versioning)) |
| `ARIES_VLLM_CCL_MODE` | `chunked` | `chunked` (fast) or `fallback` (slow, no SYCL kernels); see [All-reduce](#all-reduce) |
| `ARIES_VLLM_AR_CHUNK_BYTES` | `4194304` | Slice size for chunked mode |
| `ZE_AFFINITY_MASK` | `0,1` | Level Zero indices of the two B60s; change if the container also sees an iGPU |
| `MEM_LIMIT` | `20g` | Container RAM cap (compose) |
| `TP_SIZE` | `2` | Tensor-parallel size |
| `ENFORCE_EAGER` | `1` | `0` allows graph capture. Also set `VLLM_XPU_ENABLE_XPU_GRAPH=1`. Unsafe for hybrid models |
| `GENERATION_CONFIG` | — | Server-side sampling defaults as JSON (`--override-generation-config`), merged over the checkpoint's `generation_config.json`; requests can still override |
| `VLLM_EXTRA_ARGS` | — | Extra `vllm serve` flags, space-separated, no quoting |

Arguments after the image name in `docker run` are appended to `vllm serve` too.

### Model settings (presets)

| Variable | `vllm serve` flag |
|---|---|
| `MODEL_DIR` | checkpoint path `/models/$MODEL_DIR` (or `MODEL_PATH` for a full path) |
| `SERVED_MODEL_NAME` | `--served-model-name` |
| `MAX_MODEL_LEN` | `--max-model-len` |
| `GPU_MEMORY_UTIL` | `--gpu-memory-utilization` |
| `MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` |
| `PREFIX_CACHING` | `1` → `--enable-prefix-caching`, `0` → `--no-enable-prefix-caching` |
| `SPECULATIVE_CONFIG` | `--speculative-config` (JSON) |
| `LIMIT_MM_PER_PROMPT` | `--limit-mm-per-prompt` (JSON) |
| `TOOL_CALL_PARSER` | `--enable-auto-tool-choice --tool-call-parser` |
| `REASONING_PARSER` | `--reasoning-parser` |

### Adding a model

1. Copy `models/qwen3.8-27b.env` to `models/<name>.env` and edit it. Comment each non-obvious
   value with why it's set.
2. `MODEL_PRESET=<name>` in `.env`, `make up`, `make smoke`, `make bench LABEL=<name>`.
3. Read the smoke output text, not just the speed: fast garbage is a failure mode here (see below).
4. Contributions welcome: a preset plus its smoke/bench numbers on your hardware.

Use a pre-quantized checkpoint (FP8, INT4, ...). Online quantization (`--quantization fp8` on a
BF16 checkpoint) holds the full-precision weights in host RAM while loading: llm-scaler #672
reports 95–99% of 64 GB used for a 27B model.

## All-reduce

With TP=2 every layer ends in an all-reduce across the cards: about 128 small ones (~10 KB) per
decoded token for a 5120-wide model and ~20 MB ones per 2048-token prefill chunk. Their latency
over PCIe sets most of the speed.

- **oneCCL's SYCL-kernel all-reduce** is fast but **fails above 4 MB** on this stack
  (`zeMemOpenIpcHandle ZE_RESULT_ERROR_INVALID_ARGUMENT`), i.e. on every prefill chunk.
  `CCL_ZE_IPC_EXCHANGE=drmfd`, `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` and the
  `CCL_SYCL_ALLREDUCE_*_THRESHOLD` knobs don't help; `CCL_SYCL_ALLREDUCE_TMP_BUF=1` hangs.
- **`fallback`** (`CCL_ENABLE_SYCL_KERNELS=0`) works at all sizes but is ~5x slower for
  decode-sized calls and ~2.7x slower for prefill-sized ones.
- **`chunked`** (default) keeps SYCL kernels on and splits tensors over 4 MB into ≤4 MB slices.
  `docker/pypatch/sitecustomize.py` wraps both `torch.distributed.all_reduce` and vLLM's
  `XpuCommunicator.all_reduce` (which calls the process group directly). `make ccl-check` tests
  both paths at decode and prefill sizes.

If chunked mode misbehaves on your box, set `ARIES_VLLM_CCL_MODE=fallback` and compare with
`make ccl-check MODE=fallback`.

## Network exposure

The container uses host networking, and the server binds `127.0.0.1` by default: only processes on
the host itself can reach it. (Host networking means the container's loopback is the host's, so
every local user and process can.)

**Don't just set `HOST=0.0.0.0`.** vLLM's API key only protects `/v1/*` (and a few other
prefixes). Other routes answer anyone who can reach the port, without the key, as vLLM's
[security docs](https://docs.vllm.ai/en/latest/usage/security/) state:

- `POST /invocations`: full chat/completion inference (SageMaker-compatible), same as `/v1`
- `/tokenize`, `/detokenize`, `/version`, `/load`, `/ping`
- operational endpoints such as `/pause`, `/resume`, `/abort_requests` (when enabled)

On `0.0.0.0` that means anyone on your network can use the model and GPUs, or disrupt serving,
without the key. The entrypoint prints a warning when `HOST` isn't loopback.

**To serve other machines, put a reverse proxy in front** (nginx, Caddy, Traefik, ...), keep
`HOST=127.0.0.1`, and have the proxy:

- forward only `/v1/` (and `/health`, `/metrics` if your monitoring needs them); return 404 for
  everything else;
- terminate TLS if clients are off-host (the API key is a bearer token);
- disable response buffering for `/v1/` (streaming responses are server-sent events), e.g.
  nginx `proxy_buffering off;` with long `proxy_read_timeout`.

`HOST=0.0.0.0` without a proxy is acceptable only on a network where everyone may use the model
anyway.

## Why each fixed setting

| Setting | Where | Why |
|---|---|---|
| `--tensor-parallel-size 2` | entrypoint | both cards compute every token. Pipeline parallelism fails for Qwen3.8-27B in this vLLM (`SupportsPP` not implemented) |
| `--enforce-eager`, `VLLM_XPU_ENABLE_XPU_GRAPH=0` | entrypoint, Dockerfile | graph capture races on Gated DeltaNet state → garbage output ([llm-scaler #672](https://github.com/intel/llm-scaler/issues/672)). Costs kernel-launch overhead |
| `--disable-custom-all-reduce` | entrypoint | no XPU custom all-reduce in this stack; oneCCL does it |
| `CCL_ATL_TRANSPORT=ofi` | Dockerfile | the default `mpi` transport segfaults in `MPI_Init` in this image |
| `CCL_TOPO_P2P_ACCESS=0`, `CCL_ZE_IPC_EXCHANGE=sockets` | Dockerfile | in every validated run; not proven necessary |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Dockerfile | required for XPU workers |
| `/dev/dri` + `/dev/dri/by-path` mount | compose | without `by-path` the runtime can't map the GPUs |
| `ipc: host`, `shm_size: 16g` | compose | oneCCL IPC between TP workers; shm is a ceiling, not an allocation |
| `privileged: true` | compose | in every validated run; running without it is untested |
| `mem_limit: 20g` | compose | TP workers use ~5 GB RSS each and can grow (llm-scaler #742); the cap kills only this container |

## Troubleshooting

- **Garbled output (word salad) in long sessions.** Seen once with fp8 KV cache (uncalibrated
  scales) + MTP + prefix caching, ~1.5 h into an agent session; not seen with `auto` KV. The
  tell is MTP acceptance collapsing (normally 60–95%, 4–6% when broken). Watch
  `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total` on
  `/metrics` and alert below ~15%. Startup warnings `Using uncalibrated q_scale 1.0` mean fp8 KV
  is on.
- **`make up` exits with the logs**: the server died during start-up. `zeMemOpenIpcHandle` in
  the log means an all-reduce over 4 MB ran unchunked: check `make ccl-check`, or use
  `ARIES_VLLM_CCL_MODE=fallback`.
- **No GPUs found**: check `/dev/dri/by-path` is mounted and `ZE_AFFINITY_MASK` matches
  (`make shell`, then `sycl-ls`).
- **Port in use**: only one server can own both GPUs and the port. Stop llama.cpp or other
  servers first.
- **Busy CPU while generating**: TP workers ~150% CPU each plus the engine ~100% (shared-memory
  queue and oneCCL polling). It drops at idle.
- **Swap storm while loading**: vLLM reads the checkpoint through the page cache; with high
  swappiness the kernel swaps out idle pages. Lower swappiness or disable swap on the host.
- **Slow agent turns despite prefix caching**: agent tools that run side jobs (memory,
  summarizers) queue behind main turns on a single-user server. Point them at a smaller model.
- `/metrics` and `/health` don't need the key; only `/v1/*` does.
- llama-benchy's "cached" rows were full re-prefills on both tested models; `make smoke` measures
  the real cached return.

## Versioning

Aries vLLM has its own [semver](https://semver.org) version (`VERSION`; pre-1.0, so `.env`
variables may still change in minor releases). The upstream base is pinned in the Makefile by
tag **and digest**, so a re-pushed upstream tag can't change a build. Image tags carry both:

| Tag | Meaning |
|---|---|
| `0.1.0-llmscaler0.26.0-b2` | Aries vLLM 0.1.0 on `intel/llm-scaler-vllm:0.26.0-b2`. Immutable: never re-pushed. Pin this |
| `latest` | newest validated release |

`make version` prints the current version, base and tag; the server logs its version at start-up
(`[aries-vllm] 0.1.0 preset=...`).

Bumps: **minor** for new presets, features or a new base image (and, before 1.0, breaking `.env`
changes); **patch** for fixes and docs; **major** for breaking changes after 1.0.

### Releasing

A release (new version, new base image, or any change to the all-reduce patch or entrypoint) is
published only after it passes, on two B60s:

1. `make build`, then `make ccl-check`: every row `ok`, both `dist` and `vllm` paths.
2. `make up`, `make smoke` twice: read the generated text, compare the numbers with `docs/results.md`.
3. `make bench`, plus a long agent session (≥1 h) watching MTP acceptance for a new base image.
4. Results rows in `docs/results.md` name the version they were measured on.
5. Bump `VERSION`, commit, `git tag v<version>`, then:

```bash
docker login                     # Docker Hub access token
make push                        # refuses on a dirty tree, an untagged HEAD, or an existing tag
```

To build under your own namespace: `make build ARIES_VLLM_IMAGE=<you>/aries-vllm`. A new base
image means updating `LLM_SCALER_TAG` and `LLM_SCALER_DIGEST` in the Makefile
(`docker buildx imagetools inspect intel/llm-scaler-vllm:<tag>` shows the digest).

## License

MIT for this repository. The image is built on `intel/llm-scaler-vllm`, which contains vLLM
(Apache-2.0) and Intel components under their own licenses; those apply to the image's layers.

## References

- [LLM Scaler vLLM README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md)
- [llm-scaler #672](https://github.com/intel/llm-scaler/issues/672): hybrid Qwen models on Arc Pro, graph-capture bugs
