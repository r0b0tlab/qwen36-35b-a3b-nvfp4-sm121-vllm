#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/llm/nvfp4/nvidia/Qwen3.6-35B-A3B-NVFP4}"
IMAGE="${IMAGE:-qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp}"
CONTAINER="${CONTAINER:-qwen36-nvfp4-vllm-v025-mtp}"
PORT="${PORT:-18080}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/qwen36-nvfp4-vllm-v025}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":2,\"moe_backend\":\"triton\"}}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "${MODEL_HOST}/config.json" ]] || { echo "ERROR: MODEL_HOST must contain config.json: ${MODEL_HOST}" >&2; exit 2; }
if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "ERROR: container already exists; inspect and remove it explicitly: ${CONTAINER}" >&2
  exit 2
fi
args=(
  docker run -d
  --name "${CONTAINER}"
  --gpus all
  --network host
  --ipc=host
  --shm-size=32g
  --ulimit memlock=-1:-1
  --cap-add=IPC_LOCK
  -e "PORT=${PORT}"
  -e MODEL_ID=/models/model
  -e SERVED_MODEL_NAME=Qwen3.6-35B-A3B-NVFP4
  -e KV_CACHE_DTYPE=fp8
  -e ATTENTION_BACKEND=flashinfer
  -e MOE_BACKEND=flashinfer_b12x
  -e LINEAR_BACKEND=flashinfer_cutlass
  -e QUANTIZATION=modelopt_mixed
  -e "SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG}"
  -e "MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}"
  -e "MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}"
  -e "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}"
  -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}"
  -e LANGUAGE_MODEL_ONLY=1
  -e MAX_JOBS=6
  -e FLASHINFER_NVCC_THREADS=2
  -v "${MODEL_HOST}:/models/model:ro"
  -v "${CACHE_ROOT}/huggingface:/root/.cache/huggingface"
  -v "${CACHE_ROOT}/flashinfer:/root/.cache/flashinfer"
  -v "${CACHE_ROOT}/vllm:/root/.cache/vllm"
  "${IMAGE}"
)
printf 'command:'
printf ' %q' "${args[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi
mkdir -p "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/flashinfer" "${CACHE_ROOT}/vllm"
"${args[@]}"
echo "container=${CONTAINER} port=${PORT}"
