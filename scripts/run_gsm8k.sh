#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-18080}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$ROOT/benchmarks/runs/$RUN_ID/gsm8k}"
PHASE_FILE="${PHASE_FILE:-$ROOT/benchmarks/runs/$RUN_ID/phase.txt}"
mkdir -p "$OUT/raw"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/llm/nvfp4/nvidia/Qwen3.6-35B-A3B-NVFP4}"

"$ROOT/.venv-bench/bin/python" -m lm_eval \
  --model local-chat-completions \
  --model_args "model=Qwen3.6-35B-A3B-NVFP4,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=4,max_retries=3,tokenized_requests=False,tokenizer=$MODEL_HOST" \
  --tasks gsm8k \
  --num_fewshot 0 \
  --apply_chat_template \
  --gen_kwargs '{"max_gen_toks":2048,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  --log_samples \
  --output_path "$OUT/raw" \
  2>&1 | tee "$OUT/run.log"

"$ROOT/.venv-bench/bin/python" "$ROOT/scripts/normalize_gsm8k.py" "$OUT/raw" "$OUT/results.json"
printf 'gsm8k-complete\n' > "$PHASE_FILE"
