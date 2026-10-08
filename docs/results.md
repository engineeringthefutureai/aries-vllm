# Reference results

Measured on the reference system (see README, "Hardware"): 2x Arc Pro B60 24 GB, card 1 on
PCIe Gen5 x8, card 2 on **Gen4 x4** (both CPU-attached, no chipset hop), Core Ultra 5 250K Plus, 32 GB RAM (30 GiB usable),
`intel/llm-scaler-vllm:0.26.0-b2`, Aries vLLM 0.1.0 (rows measured during its development).
Tokens per second unless stated. Single request.

`make smoke` measures warm decode (Fibonacci prompt, code), cold prefill of a ~15K-token prompt and
the return to that cached prompt. `make bench` runs llama-benchy (pp2048/tg128 at depth 0 / 8K / 26K).

## Qwen3.8-27B-FP8, TP=2

Preset `models/qwen3.8-27b.env` unless noted (MTP 2, auto KV, prefix caching, 2048 batched tokens,
98K context).

| All-reduce | Config change | Cold prefill @0 / 8K / 26K | Decode @0 / 8K / 26K | Smoke: decode / prefill @15K / cached return | Notes |
|---|---|---|---|---|---|
| **chunked** | — | **1164–1186 / 1211–1216 / 1162–1163** | **33.0–35.7 / 34.5–35.7 / 34.8–35.5** (peak 37–46) | 42–44 / 1200–1206 / 1.7 s | two runs; decode varies with MTP acceptance (88% on code) |
| fallback | — | 707 / 710 / 693 | 26.6 / 23.9 / 25.7 | 31–32 / ~700 / — | ~26–28 tok/s and 72% acceptance in a long agent session; clean to ~89K context |
| fallback | MTP off, 32K ctx | — | — | 14.7 / ~700 / — | |
| fallback | MTP off, 8192 batched, 32K ctx | — | — | — / 690 / 1.6 s | 8192 vs 2048 batched: no prefill gain, +0.7 GB VRAM/card |
| fallback | fp8 KV, 8192 batched, 131K ctx | ~520 @85K | ~25 | — / — / 4–5 s | **Garbled output** ~1.5 h into an agent session at ~25K context; MTP acceptance fell to 4–6%. Not recommended |

Without prefix caching (vLLM's default for hybrid models) the 15K cached return is a full
re-prefill: ~41 s.

## llama.cpp on the same hardware, for comparison

SYCL backend, layer split across both cards, Qwen3.8-27B UD-Q6_K_XL + MTP (n-max 3), q8_0 KV,
131K context.

| Cold prefill @8K / 26K / 85–100K | Decode @0 / 26K | Cached return @15K | Wall power generating |
|---|---|---|---|
| 366 / 367 / 290–300 | 22.2 / 19.7 | 1.5–2 s | ~215 W (vLLM TP: ~337 W; idle ~75 W) |

## All-reduce micro-benchmark (`make ccl-check`)

Server stopped, both GPUs idle, bf16, `vllm` path (`XpuCommunicator.all_reduce`; the `dist` path
is within noise of it). Decode-sized = one token of a 5120-wide model.

| Size (bf16 elements) | fallback (SYCL kernels off) | raw SYCL kernels | chunked (≤4 MB + SYCL kernels) |
|---|---|---|---|
| 5120 (decode) | 73 µs | 15 µs | 14 µs |
| 1M (2 MB) | 1.4 ms | 0.59 ms | 0.58 ms |
| 2048×5120 (prefill chunk, 20 MB) | 14.6 ms | FAIL (zeMemOpenIpcHandle) | 5.8 ms |
| 16M (32 MB) | 26.8 ms | FAIL | 9.2 ms |
| 64M (128 MB) | 108 ms | FAIL | 37 ms |
