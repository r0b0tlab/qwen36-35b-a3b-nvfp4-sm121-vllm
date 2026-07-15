#!/usr/bin/env bash
# Launch an isolated Qwen3.6 maximum-context profile. Never removes containers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:?usage: launch_max_context.sh ar|mtp [max_model_len]}"
MAXLEN="${2:-262144}"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/llm/nvfp4/nvidia/Qwen3.6-35B-A3B-NVFP4}"
IMAGE="${QWEN_MAXCTX_IMAGE_OVERRIDE:-qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp}"
PORT="${PORT:-18080}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-6G}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/qwen36-nvfp4-vllm-v025}"
case "$PROFILE" in
  ar) SPEC=''; CONTAINER="${CONTAINER:-qwen36-maxctx-ar-${MAXLEN}}" ;;
  mtp) SPEC='{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}'; CONTAINER="${CONTAINER:-qwen36-maxctx-mtp2-${MAXLEN}}" ;;
  *) echo "invalid profile: $PROFILE" >&2; exit 2 ;;
esac
[[ -f "$MODEL_HOST/config.json" && -f "$MODEL_HOST/model.safetensors.index.json" ]] || { echo "model admission failed" >&2; exit 2; }
docker container inspect "$CONTAINER" >/dev/null 2>&1 && { echo "container exists: $CONTAINER" >&2; exit 2; }
ss -ltn "( sport = :$PORT )" | grep -q LISTEN && { echo "port busy: $PORT" >&2; exit 2; }
mkdir -p "$CACHE_ROOT"/{huggingface,flashinfer,vllm}
args=(docker run -d --name "$CONTAINER" --gpus all --network host --ipc=host --shm-size=32g
  --ulimit memlock=-1:-1 --cap-add=IPC_LOCK
  -e "PORT=$PORT" -e MODEL_ID=/models/model -e SERVED_MODEL_NAME=Qwen3.6-35B-A3B-NVFP4
  -e KV_CACHE_DTYPE=fp8 -e ATTENTION_BACKEND=flashinfer -e MOE_BACKEND=flashinfer_b12x
  -e LINEAR_BACKEND=flashinfer_cutlass -e QUANTIZATION=modelopt_mixed
  -e "SPECULATIVE_CONFIG=$SPEC" -e "MAX_MODEL_LEN=$MAXLEN" -e "MAX_NUM_SEQS=$MAX_NUM_SEQS"
  -e "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS" -e "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
  -e "KV_CACHE_MEMORY_BYTES=$KV_CACHE_MEMORY_BYTES" -e "EXTRA_ARGS=--kv-cache-memory-bytes $KV_CACHE_MEMORY_BYTES"
  -e LANGUAGE_MODEL_ONLY=1 -e MAX_JOBS=6 -e FLASHINFER_NVCC_THREADS=2
  -v "$MODEL_HOST:/models/model:ro"
  -v "$CACHE_ROOT/huggingface:/root/.cache/huggingface"
  -v "$CACHE_ROOT/flashinfer:/root/.cache/flashinfer"
  -v "$CACHE_ROOT/vllm:/root/.cache/vllm"
  "$IMAGE")
printf 'command:' >&2; printf ' %q' "${args[@]}" >&2; printf '\n' >&2
"${args[@]}"
printf 'container=%s profile=%s max_model_len=%s port=%s\n' "$CONTAINER" "$PROFILE" "$MAXLEN" "$PORT"
