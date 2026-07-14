#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?Usage: scripts/finalize_release.sh RUN_ID}"
RUN="${ROOT}/benchmarks/runs/${RUN_ID}"
for required in \
  "${RUN}/evidence-reuse.json" \
  "${RUN}/equivalence/summary.json" \
  "${RUN}/evidence/runtime-audit.json" \
  "${RUN}/evidence/w4a16-input-scale-audit-clean.json" \
  "${RUN}/runtime-manifest.json" \
  "${RUN}/gsm8k/results.json" \
  "${RUN}/concurrency/summary.json"; do
  [[ -s "${required}" ]] || { echo "Missing release input: ${required}" >&2; exit 2; }
done
mkdir -p "${ROOT}/publication/html" "${ROOT}/docs"
python3 "${ROOT}/scripts/generate_report.py" "${RUN}" "${ROOT}/publication/html/index.html"
cp "${ROOT}/publication/html/index.html" "${ROOT}/docs/index.html"
python3 "${ROOT}/scripts/verify_release.py" "${RUN}" "${ROOT}/publication/html/index.html"
python3 "${ROOT}/scripts/write_manifest.py" "${RUN}"
python3 "${ROOT}/scripts/public_safety_scan.py"
printf 'Release artifacts finalized: %s\n' "${RUN}"
