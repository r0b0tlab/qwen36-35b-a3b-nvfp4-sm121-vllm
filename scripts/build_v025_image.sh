#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp}"
BASE_MANIFEST="ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121@sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5"

docker manifest inspect "${BASE_MANIFEST}" >/dev/null
docker build --pull --file "${ROOT}/Dockerfile.v025" --tag "${IMAGE}" "${ROOT}"
docker image inspect "${IMAGE}" --format 'image_id={{.Id}} labels={{json .Config.Labels}}'
