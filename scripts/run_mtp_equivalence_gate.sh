#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?usage: scripts/run_mtp_equivalence_gate.sh RUN_ID}"
RUN="${ROOT}/benchmarks/runs/${RUN_ID}"
PORT="${PORT:-18080}"
CONTAINER="${CONTAINER:-qwen36-nvfp4-vllm-v025-mtp}"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/llm/nvfp4/nvidia/Qwen3.6-35B-A3B-NVFP4}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
[[ -s "${RUN}/evidence-reuse.json" ]] || { echo "missing evidence-reuse.json" >&2; exit 2; }
mkdir -p "${RUN}/equivalence" "${RUN}/evidence"
printf 'RUNNING\n' > "${RUN}/EQUIVALENCE_STATUS"
cleanup() {
  rc=$?
  docker logs "${CONTAINER}" > "${RUN}/equivalence/server.log" 2>&1 || true
  docker stop --time 20 "${CONTAINER}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER}" >/dev/null 2>&1 || true
  if (( rc == 0 )); then printf 'COMPLETE\n' > "${RUN}/EQUIVALENCE_STATUS"; else printf 'FAILED exit=%s\n' "$rc" > "${RUN}/EQUIVALENCE_STATUS"; fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

IMAGE="${IMAGE:-qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp}" CONTAINER="${CONTAINER}" PORT="${PORT}" MODEL_HOST="${MODEL_HOST}" bash "${ROOT}/scripts/launch.sh"
for _ in $(seq 1 "${STARTUP_TIMEOUT}"); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  state="$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "${CONTAINER}" 2>/dev/null || true)"
  if [[ "${state}" == exited* ]]; then
    echo "container exited during startup: ${state}" >&2
    exit 3
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

docker logs "${CONTAINER}" > "${RUN}/equivalence/server-live.log" 2>&1
python3 - "${RUN}/equivalence/server-live.log" "${RUN}/evidence/runtime-audit.json" <<'PY'
import json, sys
from pathlib import Path
text=Path(sys.argv[1]).read_text(errors="replace")
decoder=json.JSONDecoder()
audit=None
for index,char in enumerate(text):
    if char != "{":
        continue
    try:
        candidate,_=decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(candidate,dict) and candidate.get("status") == "PASS" and candidate.get("provenance") == "r0b0tlab-qwen36-nvidia-v025-mtp":
        audit=candidate
        break
if audit is None:
    raise SystemExit("startup runtime audit PASS payload not found")
Path(sys.argv[2]).write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
PY
cp "${RUN}/evidence/w4a16-input-scale-audit.json" "${RUN}/evidence/w4a16-input-scale-audit-clean.json"
python3 "${ROOT}/scripts/run_semantic_gate.py" --base-url "http://127.0.0.1:${PORT}" --output "${RUN}/equivalence/semantic.json"
python3 "${ROOT}/scripts/run_long_generation.py" --base-url "http://127.0.0.1:${PORT}" --output "${RUN}/equivalence/long-generation.json"

for c in 1 32; do
  prompts=$((c * 2)); (( prompts < 8 )) && prompts=8
  docker exec "${CONTAINER}" /usr/local/bin/vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:${PORT}" \
    --endpoint /v1/chat/completions \
    --model Qwen3.6-35B-A3B-NVFP4 \
    --tokenizer /models/model \
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
    --result-dir /tmp \
    --result-filename "equivalence-c${c}.json" \
    2>&1 | tee "${RUN}/equivalence/c${c}.log"
  docker cp "${CONTAINER}:/tmp/equivalence-c${c}.json" "${RUN}/equivalence/c${c}.json"
done
CONTAINER="${CONTAINER}" PORT="${PORT}" python3 "${ROOT}/scripts/capture_release.py" "${RUN}" > "${RUN}/equivalence/runtime-manifest.stdout.json"
python3 - "${RUN}" <<'PY'
import json, sys
from pathlib import Path
r=Path(sys.argv[1])
source=json.loads((r/'concurrency/summary.json').read_text())
clean={}
canary_ok={}
for c in (1,32):
    p=json.loads((r/f'equivalence/c{c}.json').read_text())
    clean[c]=float(p['output_throughput'])
    canary_ok[c]=int(p.get('completed') or 0) > 0 and int(p.get('failed') or 0) == 0 and clean[c] > 0
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
 'c1_functional_canary': canary_ok[1],
 'c32_functional_canary': canary_ok[32],
 'mtp_acceptance_positive': 0 < float(runtime.get('mtp_acceptance_rate') or 0) <= 1,
}
payload={'reference_output_tok_s':{str(k):v for k,v in reference.items() if k in clean},'clean_canary_output_tok_s':{str(k):v for k,v in clean.items()},'ratios':ratios,'checks':checks,'passed':all(checks.values()),'canary_policy':'functional packaging check only; publication throughput comes from hash-verified three-repetition source evidence'}
(r/'equivalence/summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
if not payload['passed']:
    raise SystemExit(json.dumps(payload,indent=2))
print(json.dumps(payload,indent=2))
PY
