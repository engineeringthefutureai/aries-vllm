# Aries vLLM: tensor parallelism (TP=2) on 2x Intel Arc Pro B60.
# Thin layer over Intel's LLM Scaler vLLM image: the env and patches TP needs on this
# hardware are baked in; model settings come from a preset (MODEL_PRESET, models/*.env) or env.
# Build with `make build`: it passes the pinned base (tag@digest) and the version from VERSION.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG LLM_SCALER_DIGEST=""
ARG ARIES_VLLM_VERSION=dev
ARG SOURCE_URL=https://github.com/engineeringthefutureai/aries-vllm

LABEL org.opencontainers.image.title="aries-vllm" \
      org.opencontainers.image.description="vLLM tensor parallelism on 2x Intel Arc Pro B60" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${ARIES_VLLM_VERSION}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.base.digest="${LLM_SCALER_DIGEST}"
ENV ARIES_VLLM_VERSION=${ARIES_VLLM_VERSION}

# --- vLLM on XPU -----------------------------------------------------------
# spawn: required for XPU multiproc workers.
# XPU graph off (with --enforce-eager from the entrypoint): graph capture races on the
#   recurrent state of hybrid (Gated DeltaNet) models -> garbage output (llm-scaler #672).
#   Untested on other models; override both only after validating.
# ZE_AFFINITY_MASK: the two B60s. If the container also sees an iGPU, indices shift;
#   check with `make shell` + `sycl-ls` and override in .env.
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_XPU_ENABLE_XPU_GRAPH=0 \
    ZE_AFFINITY_MASK=0,1

# --- oneCCL (cross-card all-reduce) ----------------------------------------
# ofi: the default mpi transport segfaults in MPI_Init inside this image.
# P2P off + sockets IPC exchange: present in every validated run; not proven necessary.
# CCL_ENABLE_SYCL_KERNELS is set by entrypoint.sh from ARIES_VLLM_CCL_MODE.
ENV CCL_ATL_TRANSPORT=ofi \
    CCL_TOPO_P2P_ACCESS=0 \
    CCL_ZE_IPC_EXCHANGE=sockets

# Chunked all-reduce patch (sitecustomize). Only active when entrypoint.sh puts
# /opt/aries-vllm/pypatch on PYTHONPATH (ARIES_VLLM_CCL_MODE=chunked).
COPY docker/pypatch/ /opt/aries-vllm/pypatch/
COPY docker/entrypoint.sh /opt/aries-vllm/entrypoint.sh
COPY tests/ /opt/aries-vllm/tests/
COPY models/ /opt/aries-vllm/models/
COPY LICENSE /opt/aries-vllm/LICENSE
RUN chmod +x /opt/aries-vllm/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/opt/aries-vllm/entrypoint.sh"]
