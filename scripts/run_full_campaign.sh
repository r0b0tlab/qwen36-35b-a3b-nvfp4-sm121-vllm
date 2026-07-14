#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?Usage: scripts/run_full_campaign.sh RUN_ID}"
RUN="${ROOT}/benchmarks/runs/${RUN_ID}"
PORT="${PORT:-18080}"
CONTAINER="${CONTAINER:-qwen36-nvfp4-vllm}"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/llm/nvfp4/nvidia/Qwen3.6-35B-A3B-NVFP4}"
BENCH_PY="${BENCH_PY:-${ROOT}/.venv-bench/bin/python}"

[[ ! -e "${RUN}" ]] || { echo "run directory already exists: ${RUN}" >&2; exit 2; }
[[ -x "${BENCH_PY}" ]] || { echo "missing benchmark Python: ${BENCH_PY}" >&2; exit 2; }
[[ "$(docker inspect -f '{{.State.Status}}' "${CONTAINER}" 2>/dev/null || true)" == running ]] || {
  echo "release container is not running: ${CONTAINER}" >&2
  exit 2
}
mkdir -p "${RUN}/evidence" "${RUN}/semantic_gate" "${RUN}/full"
printf 'preflight\n' > "${RUN}/phase.txt"
printf 'RUNNING\n' > "${RUN}/STATUS"

telemetry_pid=""
cleanup() {
  rc=$?
  if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
    kill -TERM "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" 2>/dev/null || true
  fi
  if (( rc == 0 )); then
    printf 'COMPLETE\n' > "${RUN}/STATUS"
    if [[ "${STOP_CONTAINER_ON_SUCCESS:-1}" == "1" ]]; then
      docker stop --time 30 "${CONTAINER}" >/dev/null
      docker rm "${CONTAINER}" >/dev/null
    fi
  else
    printf 'FAILED exit=%s\n' "${rc}" > "${RUN}/STATUS"
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

python3 "${ROOT}/scripts/telemetry.py" "${RUN}/telemetry.jsonl" \
  --interval 2 --phase-file "${RUN}/phase.txt" &
telemetry_pid=$!
printf '%s\n' "${telemetry_pid}" > "${RUN}/telemetry.pid"

printf 'runtime-audit\n' > "${RUN}/phase.txt"
docker exec "${CONTAINER}" python3 /opt/r0b0tlab/audit_runtime_v025.py \
  > "${RUN}/evidence/runtime-audit.json"
"${BENCH_PY}" "${ROOT}/scripts/audit_w4a16_input_scales.py" "${MODEL_HOST}" \
  --output "${RUN}/evidence/w4a16-input-scale-audit.json" >/dev/null
curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" \
  > "${RUN}/evidence/models-preflight.json"
logs="$(docker logs "${CONTAINER}" 2>&1)"
if printf '%s' "${logs}" | grep -qiE "Marlin kernel|marlin\.py|Using .MARLIN.|W4A16.*weight.only|falling back.*emulation|Using .EMULATION.|missing.*input.scale"; then
  echo "forbidden runtime marker before campaign" >&2
  exit 3
fi
for marker in R0B0TLAB_NATIVE_W4A4_FROM_W4A16 FlashInferFP8ScaledMMLinearKernel FlashInferCutlassNvFp4LinearKernel FLASHINFER_B12X PIECEWISE; do
  grep -q "${marker}" <<<"${logs}" || { echo "missing runtime marker: ${marker}" >&2; exit 3; }
done

printf 'semantic-gate\n' > "${RUN}/phase.txt"
python3 "${ROOT}/scripts/run_semantic_gate.py" \
  --base-url "http://127.0.0.1:${PORT}" --output "${RUN}/semantic_gate/results.json"
printf 'long-generation\n' > "${RUN}/phase.txt"
python3 "${ROOT}/scripts/run_long_generation.py" \
  --base-url "http://127.0.0.1:${PORT}" --output "${RUN}/long_generation.json"
RUN_ID="${RUN_ID}" PORT="${PORT}" BENCH_PY="${BENCH_PY}" MODEL_HOST="${MODEL_HOST}" PHASE_FILE="${RUN}/phase.txt" \
  bash "${ROOT}/scripts/run_gsm8k.sh"
RUN_ID="${RUN_ID}" PORT="${PORT}" CONTAINER="${CONTAINER}" PHASE_FILE="${RUN}/phase.txt" \
  bash "${ROOT}/scripts/run_concurrency.sh"
RUN_ID="${RUN_ID}" PORT="${PORT}" PHASE_FILE="${RUN}/phase.txt" \
  bash "${ROOT}/scripts/run_llama_benchy.sh"
printf 'durability\n' > "${RUN}/phase.txt"
python3 "${ROOT}/scripts/run_durability.py" \
  --base-url "http://127.0.0.1:${PORT}" --output "${RUN}/full/durability.jsonl" --requests 250

printf 'finalize-release\n' > "${RUN}/phase.txt"
PORT="${PORT}" CONTAINER="${CONTAINER}" bash "${ROOT}/scripts/finalize_release.sh" "${RUN_ID}"
printf 'campaign-complete\n' > "${RUN}/phase.txt"
echo "FULL_CAMPAIGN_COMPLETE run=${RUN}"
