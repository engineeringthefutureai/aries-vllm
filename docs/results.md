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

## Gemma 4 26B A4B FP8, TP=2

Preset `models/gemma-4-26b-a4b.env` (RedHatAI FP8-dynamic checkpoint, 262K context, prefix caching,
no speculative decoding, 2048 batched tokens). With the FP8 checkpoint, MoE runs on vLLM's
**Triton** FP8 kernels (the XPU FP8 MoE kernels don't take per-channel scales; no tuned MoE config
for this GPU); weights 12.6 GiB/card, KV pool 651K tokens. Attention runs on Triton in every row
(XPU Flash Attention doesn't support Gemma's prefix-LM attention). Each row below changes one
thing from the first.

| All-reduce | Config change | Cold prefill @0 / 8K / 26K | Decode @0 / 8K / 26K | Smoke: decode / prefill @17K / cached return | Notes |
|---|---|---|---|---|---|
| chunked | — | 643 / 420 / 231 | 27.0 / 25.9 / 20.8 | 26.8 / 324 / 0.25–0.96 s | output read: clean; prefill falls with depth (attention, likely the Triton fallback) |
| chunked | `ENFORCE_EAGER=0` + `VLLM_XPU_ENABLE_XPU_GRAPH=1` | — | — | — | **start-up crash** in graph capture: compiled Gemma 4 code calls `.item()` (`wait method cannot be used for an event associated with a command graph`) |
| chunked | `ENFORCE_EAGER=0` (torch.compile, no graph) | — | — | 22.8 / 319 / 0.27 s | **garbage output** (word salad, decode runs stop at 10–67 tokens). Compiler warns that vLLM's all-reduce output aliases its input. Not usable |
| chunked | + `gemma-4-26B-A4B-it-assistant` drafter, 2 tokens | 613 / 382 / 202 | 26.7 / **14.8 / 7.3** | **48** / 290 / 1.28 s | acceptance 87% on smoke (code), 66% cumulative with bench. Big win at short context, 3x slower decode at 26K |
| chunked | **QAT INT4 checkpoint** (`cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`) | **2547 / 1271 / 529** | **26.0 / 25.9 / 25.7** | 26.2 / 864 / 0.14 s | native XPU INT4 MoE + linear kernels; 7.9 GiB weights/card, KV pool 1.08M tokens; output read: clean. Quality vs FP8 not measured |
| — (one card) | QAT INT4, `TP_SIZE=1`, 131K context (262K doesn't fit one card) | 2233 / 750 / 252 | **33.9 / 33.8 / 28.7** | 33.8 / 460 / 0.16 s | vs the TP=2 INT4 row: decode +30% (TP adds ~8.5 ms/token), prefill 1.1-2x slower. Still a ~30 ms/token floor on one card (weights take ~4-5 ms to read, estimate): overhead in Gemma's code path (MoE routing kernels, Triton attention) dominates. Not a stack-wide floor: dense Devstral with native kernels spends only ~6 ms/token beyond its weight reads. KV pool 181K tokens |

llama-benchy's "cached" rows re-prefilled here too (pp256 @ 8K ttfr 18.7 s), although this model
isn't hybrid; the smoke cached return (same prompt resent) shows prefix caching works.

## Devstral Small 2 24B FP8, TP=2

Preset `models/devstral-small-2-24b.env` (Mistral's own FP8 checkpoint, dense 24B, first row at 98K context,
prefix caching, no speculative decoding, 2048 batched tokens). Native XPU kernels throughout:
`XPUW8A16FP8LinearKernel`, Flash Attention. Weights 11.7 GiB/card, KV pool 122K tokens.

| All-reduce | Config change | Cold prefill @0 / 8K / 26K | Decode @0 / 8K / 26K | Smoke: decode / prefill @15K / cached return | Notes |
|---|---|---|---|---|---|
| chunked | — | **1894 / 1945 / 1765** | 31.3 / 29.4 / 26.6 | 31.2 / 1932 / 0.13 s | output read: clean. Decode step ~32 ms, of which ~26 ms is reading 12 GiB/card (estimate): near its bandwidth ceiling |
| chunked | 131K context: `MAX_MODEL_LEN=131072`, `GPU_MEMORY_UTIL=0.95` (needed together: 131K needs ~10 GiB KV/card) | — | — | 31.2 / 1931 / 0.13 s | KV pool 137,792 tokens (10.52 GiB/card; one full-length request at a time). 116K-token prompt with a codename in its first line: recalled exactly; cold prefill 96 s (1209 tok/s avg), cached repeat 0.72 s. 262K would need fp8 KV, and the checkpoint has no KV scales |

## Model comparison (TP=2, best config each, tok/s)

| Model | Cold prefill @0 / 8K / 26K | Decode @0 / 8K / 26K | Max context | Notes |
|---|---|---|---|---|
| Qwen3.8-27B FP8 + MTP | 1164–1186 / 1211–1216 / 1162–1163 | 33.0–35.7 / 34.5–35.7 / 34.8–35.5 | 98K | MTP carries decode |
| Devstral Small 2 24B FP8 | **1894 / 1945 / 1765** | 31.3 / 29.4 / 26.6 | 131K (393K native) | no speculative decoding; native kernels |
| Gemma 4 26B A4B QAT INT4 | 2547 / 1271 / 529 | 26.0 / 25.9 / 25.7 | 262K | Triton attention fallback slows deep prefill |

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
