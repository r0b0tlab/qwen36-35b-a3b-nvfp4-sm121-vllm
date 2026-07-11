#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?Usage: scripts/run_full_campaign.sh RUN_ID}"
RUN="$ROOT/benchmarks/runs/$RUN_ID"
PORT="${PORT:-18080}"
CONTAINER="${CONTAINER:-qwen36-nvfp4-vllm}"
BASELINE="$ROOT/benchmarks/runs/fp8-native-w4a4-v1"

mkdir -p "$RUN/evidence" "$RUN/semantic_gate"
printf 'preflight\n' > "$RUN/phase.txt"
printf 'RUNNING\n' > "$RUN/STATUS"

# Seed the expanded native semantic/long-generation evidence into the campaign.
cp "$BASELINE/semantic_gate/results.json" "$RUN/semantic_gate/results.json"
cp "$BASELINE/long_generation.json" "$RUN/long_generation.json"
cp "$BASELINE/evidence/w4a16-input-scale-audit.json" "$RUN/evidence/w4a16-input-scale-audit.json"

telemetry_pid=""
cleanup() {
  rc=$?
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill -TERM "$telemetry_pid"
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if (( rc == 0 )); then
    printf 'COMPLETE\n' > "$RUN/STATUS"
  else
    printf 'FAILED exit=%s\n' "$rc" > "$RUN/STATUS"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

python3 "$ROOT/scripts/telemetry.py" "$RUN/telemetry.jsonl" \
  --interval 2 --phase-file "$RUN/phase.txt" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" > "$RUN/telemetry.pid"

# Reconfirm native server identity and reject forbidden runtime markers.
curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" > "$RUN/evidence/models-preflight.json"
logs="$(docker logs "$CONTAINER" 2>&1)"
if printf '%s' "$logs" | grep -qiE "Marlin kernel|marlin\.py|Using .MARLIN.|falling back.*emulation|Using .EMULATION."; then
  echo "forbidden runtime marker before campaign" >&2
  exit 3
fi

RUN_ID="$RUN_ID" PORT="$PORT" PHASE_FILE="$RUN/phase.txt" \
  bash "$ROOT/scripts/run_gsm8k.sh"
RUN_ID="$RUN_ID" PORT="$PORT" CONTAINER="$CONTAINER" PHASE_FILE="$RUN/phase.txt" \
  bash "$ROOT/scripts/run_concurrency.sh"
RUN_ID="$RUN_ID" PORT="$PORT" PHASE_FILE="$RUN/phase.txt" \
  bash "$ROOT/scripts/run_llama_benchy.sh"

printf 'finalize-release\n' > "$RUN/phase.txt"
PORT="$PORT" CONTAINER="$CONTAINER" bash "$ROOT/scripts/finalize_release.sh" "$RUN_ID"
printf 'campaign-complete\n' > "$RUN/phase.txt"

echo "FULL_CAMPAIGN_COMPLETE run=$RUN"
