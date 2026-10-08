"""Chunked all-reduce for vLLM tensor parallelism on 2x Intel Arc Pro B60.

oneCCL's SYCL-kernel all-reduce fails above 4 MB on this setup
(zeMemOpenIpcHandle ZE_RESULT_ERROR_INVALID_ARGUMENT) but is ~3-5x faster than the
CCL_ENABLE_SYCL_KERNELS=0 fallback. This reduces XPU tensors larger than 4 MB in <=4 MB
slices; decode-sized calls pass through unchanged. Wrapped entry points:
torch.distributed.all_reduce and vLLM's XpuCommunicator.all_reduce (which calls the
ProcessGroup directly). Loaded automatically in every Python process (incl. spawned TP
workers) when this directory is on PYTHONPATH. Idle GPUs, vs fallback: 10.5M bf16 elements
14.6 ms -> 5.8 ms; 5120 elements 73 us -> 14 us (make ccl-check).
"""
import os, sys


def _chain_system_sitecustomize():
    """Run the image's own sitecustomize (e.g. /usr/lib/python3.12/sitecustomize.py),
    which this file shadows by being first on PYTHONPATH."""
    import runpy
    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        p = os.path.join(d or ".", "sitecustomize.py")
        if os.path.abspath(os.path.dirname(p)) != here and os.path.isfile(p):
            try:
                runpy.run_path(p, run_name="_system_sitecustomize")
            except Exception as e:
                print(f"[aries-vllm] system sitecustomize {p} failed: {e}", file=sys.stderr, flush=True)
            return

_chain_system_sitecustomize()

LIMIT = int(os.environ.get("ARIES_VLLM_AR_CHUNK_BYTES", str(4 << 20)))

def _install():
    import torch
    import torch.distributed as dist
    if getattr(dist.all_reduce, "_aries_vllm_chunked", False):
        return
    orig = dist.all_reduce

    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        nbytes = tensor.numel() * tensor.element_size()
        if (async_op or nbytes <= LIMIT or tensor.device.type != "xpu"
                or not tensor.is_contiguous()):
            return orig(tensor, op=op, group=group, async_op=async_op)
        flat = tensor.view(-1)
        step = max(1, LIMIT // tensor.element_size())
        for i in range(0, flat.numel(), step):
            orig(flat[i:i + step], op=op, group=group)
        return None

    all_reduce._aries_vllm_chunked = True
    dist.all_reduce = all_reduce
    if os.environ.get("ARIES_VLLM_AR_CHUNK_QUIET") != "1":
        print(f"[aries-vllm] chunked all_reduce active (limit {LIMIT >> 20} MB), pid {os.getpid()}",
              file=sys.stderr, flush=True)


# vLLM's XpuCommunicator.all_reduce (eager path) calls device_group.allreduce() on the
# ProcessGroup directly, bypassing dist.all_reduce, so it needs its own wrapper. The
# module is patched right after it is imported (importing vLLM here would be too heavy).
_XPU_COMM = "vllm.distributed.device_communicators.xpu_communicator"


def _patch_xpu_communicator(mod):
    cls = mod.XpuCommunicator
    if getattr(cls.all_reduce, "_aries_vllm_chunked", False):
        return
    orig = cls.all_reduce

    def all_reduce(self, input_):
        nbytes = input_.numel() * input_.element_size()
        if nbytes <= LIMIT or input_.device.type != "xpu" or not input_.is_contiguous():
            return orig(self, input_)
        # orig reduces in place by default, or returns a clone; handle both per slice.
        out = input_.view(-1)
        step = max(1, LIMIT // input_.element_size())
        for i in range(0, out.numel(), step):
            sl = out[i:i + step]
            r = orig(self, sl)
            if r.data_ptr() != sl.data_ptr():
                sl.copy_(r)
        return input_

    all_reduce._aries_vllm_chunked = True
    cls.all_reduce = all_reduce
    if os.environ.get("ARIES_VLLM_AR_CHUNK_QUIET") != "1":
        print(f"[aries-vllm] chunked XpuCommunicator.all_reduce active, pid {os.getpid()}",
              file=sys.stderr, flush=True)


class _PatchOnImport:
    """Meta-path finder: when _XPU_COMM is imported, patch it after it executes."""

    def find_spec(self, name, path=None, target=None):
        if name != _XPU_COMM:
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                _patch_xpu_communicator(module)
            except Exception as e:
                print(f"[aries-vllm] XpuCommunicator patch NOT installed: {e}", file=sys.stderr,
                      flush=True)

        spec.loader.exec_module = exec_and_patch
        return spec


try:
    _install()
    sys.meta_path.insert(0, _PatchOnImport())
except Exception as e:  # never break interpreter start-up
    print(f"[aries-vllm] chunked all_reduce NOT installed: {e}", file=sys.stderr, flush=True)
