#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?usage: scripts/run_mtp_equivalence_gate.sh RUN_ID}"
RUN="${ROOT}/benchmarks/runs/${RUN_ID}"
PORT="${PORT:-18080}"
CONTAINER="${CONTAINER:-qwen36-nvfp4-vllm-v025-mtp}"
BENCH_BIN="${BENCH_BIN:-${ROOT}/.venv-bench/bin/vllm}"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/llm/nvfp4/nvidia/Qwen3.6-35B-A3B-NVFP4}"
[[ -s "${RUN}/evidence-reuse.json" ]] || { echo "missing evidence-reuse.json" >&2; exit 2; }
mkdir -p "${RUN}/equivalence" "${RUN}/evidence"
printf 'RUNNING\n' > "${RUN}/EQUIVALENCE_STATUS"
cleanup() {
  rc=$?
  docker stop --time 20 "${CONTAINER}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER}" >/dev/null 2>&1 || true
  if (( rc == 0 )); then printf 'COMPLETE\n' > "${RUN}/EQUIVALENCE_STATUS"; else printf 'FAILED exit=%s\n' "$rc" > "${RUN}/EQUIVALENCE_STATUS"; fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

IMAGE="${IMAGE:-qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp}" CONTAINER="${CONTAINER}" PORT="${PORT}" MODEL_HOST="${MODEL_HOST}" bash "${ROOT}/scripts/launch.sh"
for _ in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

docker exec "${CONTAINER}" python3 /opt/r0b0tlab/audit_runtime_v025.py > "${RUN}/evidence/runtime-audit.json"
"${ROOT}/.venv-bench/bin/python" "${ROOT}/scripts/audit_w4a16_input_scales.py" \
  "${MODEL_HOST}" --output "${RUN}/evidence/w4a16-input-scale-audit-clean.json"
python3 "${ROOT}/scripts/run_semantic_gate.py" --base-url "http://127.0.0.1:${PORT}" --output "${RUN}/equivalence/semantic.json"
python3 "${ROOT}/scripts/run_long_generation.py" --base-url "http://127.0.0.1:${PORT}" --output "${RUN}/equivalence/long-generation.json"

for c in 1 32; do
  prompts=$((c * 2)); (( prompts < 8 )) && prompts=8
  "${BENCH_BIN}" bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:${PORT}" \
    --endpoint /v1/chat/completions \
    --model Qwen3.6-35B-A3B-NVFP4 \
    --tokenizer "${MODEL_HOST}" \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 512 \
    --num-prompts "${prompts}" \
    --max-concurrency "${c}" \
    --request-rate inf \
    --seed 0 \
    --ignore-eos \
    --temperature 0 \
    --save-result \
    --result-filename "${RUN}/equivalence/c${c}.json"
done
CONTAINER="${CONTAINER}" PORT="${PORT}" python3 "${ROOT}/scripts/capture_release.py" "${RUN}" > "${RUN}/equivalence/runtime-manifest.stdout.json"
python3 - "${RUN}" <<'PY'
import json, sys
from pathlib import Path
r=Path(sys.argv[1])
source=json.loads((r/'concurrency/summary.json').read_text())
clean={}
for c in (1,32):
    p=json.loads((r/f'equivalence/c{c}.json').read_text())
    clean[c]=float(p['output_throughput'])
reference={int(x['concurrency']):float(x['output_throughput_tok_s']) for x in source['rows']}
ratios={str(c):clean[c]/reference[c] for c in clean}
semantic=json.loads((r/'equivalence/semantic.json').read_text())
long=json.loads((r/'equivalence/long-generation.json').read_text())
runtime=json.loads((r/'runtime-manifest.json').read_text())
audit=json.loads((r/'evidence/runtime-audit.json').read_text())
checks={
 'runtime_audit': audit.get('status')=='PASS' and not audit.get('failures'),
 'semantic': semantic.get('passed') is True,
 'long_generation': long.get('passed') is True,
 'c1_ratio_at_least_0_70': ratios['1'] >= .70,
 'c32_ratio_at_least_0_70': ratios['32'] >= .70,
 'mtp_acceptance_positive': 0 < float(runtime.get('mtp_acceptance_rate') or 0) <= 1,
}
payload={'reference_output_tok_s':{str(k):v for k,v in reference.items() if k in clean},'clean_canary_output_tok_s':{str(k):v for k,v in clean.items()},'ratios':ratios,'checks':checks,'passed':all(checks.values())}
(r/'equivalence/summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
if not payload['passed']:
    raise SystemExit(json.dumps(payload,indent=2))
print(json.dumps(payload,indent=2))
PY
