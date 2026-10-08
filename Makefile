# Aries vLLM: TP=2 on 2x Arc Pro B60. Run `make help`.
.PHONY: help build push pull version up down restart logs ps smoke bench ccl-check shell config
-include .env

# Versions: ours in VERSION (semver); the upstream base pinned by tag + digest. Image tags are
# <version>-llmscaler<base tag> (immutable) plus latest. See README, "Versioning".
VERSION := $(shell cat VERSION)
LLM_SCALER_TAG := 0.26.0-b2
LLM_SCALER_DIGEST := sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
BASE_IMAGE := intel/llm-scaler-vllm:$(LLM_SCALER_TAG)@$(LLM_SCALER_DIGEST)
BUILD_TAG := $(VERSION)-llmscaler$(LLM_SCALER_TAG)
ARIES_VLLM_IMAGE ?= engineeringthefuture/aries-vllm
ARIES_VLLM_TAG ?= $(BUILD_TAG)
export ARIES_VLLM_IMAGE ARIES_VLLM_TAG BASE_IMAGE LLM_SCALER_DIGEST
export ARIES_VLLM_VERSION := $(VERSION)
PORT ?= 8080
MODE ?= chunked
LABEL ?= $(shell date +%m%d)

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-11s %s\n", $$1, $$2}'
build:           ## build $(ARIES_VLLM_IMAGE):<version>-llmscaler<base> and :latest
	ARIES_VLLM_TAG=$(BUILD_TAG) docker compose build
	docker tag $(ARIES_VLLM_IMAGE):$(BUILD_TAG) $(ARIES_VLLM_IMAGE):latest
push:            ## release: push both tags (needs docker login, clean tree, git tag v$(VERSION))
	@test -z "$$(git status --porcelain)" || { echo "uncommitted changes; commit first"; exit 1; }
	@test "$$(git tag --points-at HEAD | grep -x 'v$(VERSION)')" || { echo "HEAD is not tagged v$(VERSION)"; exit 1; }
	@if docker manifest inspect $(ARIES_VLLM_IMAGE):$(BUILD_TAG) >/dev/null 2>&1; then \
	  echo "$(ARIES_VLLM_IMAGE):$(BUILD_TAG) already exists; release tags are immutable, bump VERSION"; exit 1; fi
	docker push $(ARIES_VLLM_IMAGE):$(BUILD_TAG)
	docker push $(ARIES_VLLM_IMAGE):latest
pull:            ## pull this checkout's release image from Docker Hub
	docker compose pull vllm
version:         ## print version, base image and image tag
	@echo "aries-vllm $(VERSION)  base $(BASE_IMAGE)  image $(ARIES_VLLM_IMAGE):$(ARIES_VLLM_TAG)"
up:              ## start the server and wait for /health
	@if ss -ltn "sport = :$(PORT)" | grep -q LISTEN; then echo "port $(PORT) is in use; stop the other server first"; exit 1; fi
	docker compose up -d vllm
	@echo "waiting for /health (model load + warm-up takes a few minutes)..."
	@until docker inspect -f '{{.State.Health.Status}}' aries-vllm 2>/dev/null | grep -qx healthy; do \
	  if [ "$$(docker inspect -f '{{.State.Running}}' aries-vllm 2>/dev/null)" != true ]; then docker compose logs --tail 40 vllm; exit 1; fi; sleep 5; done
	@echo "up"; docker compose logs vllm | grep -E "\[aries-vllm\]" || true
down:            ## stop and remove the container
	docker compose down
restart: down up ## down + up (picks up .env / preset changes)
logs:            ## follow server logs
	docker compose logs -f vllm
ps:              ## container status + health
	docker compose ps
smoke:           ## correctness + decode / cold prefill / cached return
	scripts/smoke.sh
bench:           ## llama-benchy cold + cached into results/ (LABEL=...)
	scripts/bench.sh $(LABEL)
ccl-check:       ## two-card all-reduce check (MODE=chunked|fallback|raw); server must be down
	scripts/ccl-check.sh $(MODE)
shell:           ## bash in a fresh container with GPUs (no server)
	docker compose run --rm --entrypoint bash vllm
config:          ## print the resolved compose config (env_file values not expanded)
	docker compose config --no-env-resolution
