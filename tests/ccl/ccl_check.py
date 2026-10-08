#!/usr/bin/env python3
"""Two-card xccl all-reduce check: correctness + timing per message size.

Runs inside the image (needs both GPUs free: stop vLLM first). If the chunking
patch is on PYTHONPATH, torch.distributed.all_reduce is the patched one, so the
same script compares modes:

  make ccl-check                     # chunked mode (SYCL kernels + patch)
  make ccl-check MODE=fallback       # CCL_ENABLE_SYCL_KERNELS=0, no patch
  make ccl-check MODE=raw            # SYCL kernels, no patch (expect FAIL > 4 MB)

Each size runs through two paths: "dist" (torch.distributed.all_reduce) and "vllm"
(vLLM's XpuCommunicator.all_reduce, which calls the ProcessGroup directly and is what
TP actually uses). The patch must cover both: a dist-only patch passes the dist rows
and still crashes vLLM at start-up.

Expected on llm-scaler-vllm 0.26.0-b2: raw fails at 10.5M bf16 elements with
zeMemOpenIpcHandle ZE_RESULT_ERROR_INVALID_ARGUMENT; fallback passes everything but is
slow (~73 us per decode-sized call vs ~15 us raw); chunked passes everything, ~3-5x faster
than fallback. A 5120-wide model's decode does ~128 all-reduces of 5120 bf16 per token.
"""
import os, sys, time, types
import torch, torch.distributed as dist, torch.multiprocessing as mp

SIZES = [5120, 1 << 20, 2048 * 5120, 1 << 24, 1 << 26]   # 5120 = one decode token (hidden size)
DTYPES = [torch.bfloat16, torch.float32]
ITERS = {5120: 1000}                                        # small sizes: more iterations


def run(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29517")
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", rank=rank, world_size=2)
    from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
    opts = dist.AllreduceOptions(); opts.reduceOp = dist.ReduceOp.SUM; opts.asyncOp = False
    comm = types.SimpleNamespace(device_group=dist.distributed_c10d._get_default_group(),
                                 _allreduce_options=opts)
    paths = {"dist": dist.all_reduce, "vllm": lambda t: XpuCommunicator.all_reduce(comm, t)}
    if rank == 0:
        print(f"CCL_ENABLE_SYCL_KERNELS={os.environ.get('CCL_ENABLE_SYCL_KERNELS')} "
              f"chunk_patch dist={getattr(dist.all_reduce, '_aries_vllm_chunked', False)} "
              f"vllm={getattr(XpuCommunicator.all_reduce, '_aries_vllm_chunked', False)}", flush=True)
    for name, ar in paths.items():
        for dt in DTYPES:
            for n in SIZES:
                tag = f"{name:4} {str(dt):15} {n:>9}"
                try:
                    t = torch.ones(n, dtype=dt, device="xpu") * (rank + 1)
                    r = ar(t); torch.xpu.synchronize()
                    ok = bool(((t if r is None else r) == 3).all().item())
                    it = ITERS.get(n, 10)
                    for _ in range(min(it, 20)): ar(t)                   # warm-up
                    torch.xpu.synchronize(); s = time.perf_counter()
                    for _ in range(it): ar(t)
                    torch.xpu.synchronize()
                    us = (time.perf_counter() - s) / it * 1e6
                    if rank == 0:
                        print(f"{'ok  ' if ok else 'BAD '} {tag} {us:10.0f} us", flush=True)
                except Exception as e:  # noqa: BLE001
                    if rank == 0:
                        print(f"FAIL {tag} {str(e)[:90]}", flush=True)
                    dist.destroy_process_group()
                    return
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, nprocs=2)
