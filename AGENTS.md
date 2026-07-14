# AGENTS.md

## Purpose

Reproduce NVIDIA's pinned Qwen3.6 35B-A3B NVFP4 checkpoint on one GB10/SM121 with correctness, native-path, provenance, and public-artifact hygiene as release gates.

## Immutable identity

- Model: `nvidia/Qwen3.6-35B-A3B-NVFP4`
- Revision: `491c2f1ea524c639598bf8fa787a93fed5a6fbce`
- Mount weights read-only. Never recalibrate, rewrite, synthesize scales, or commit model/cache files.
- Base image digest: `sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5`
- vLLM commit: `702f4814fe54fabff350d43cb753ae3e47c0c276`

## Accepted profile

```text
QUANTIZATION=modelopt_mixed
KV_CACHE_DTYPE=fp8
ATTENTION_BACKEND=flashinfer
MOE_BACKEND=flashinfer_b12x
LINEAR_BACKEND=flashinfer_cutlass
SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}
GPU_MEMORY_UTILIZATION=0.88
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=32768
```

Required effective markers: `FlashInferFP8ScaledMMLinearKernel`, `R0B0TLAB_NATIVE_W4A4_FROM_W4A16`, `FlashInferCutlassNvFp4LinearKernel`, `FLASHINFER_B12X`, FlashInfer attention, MTP K=2, and `PIECEWISE`. Any Marlin, weight-only W4A16 execution, emulation, missing-scale, or fallback marker invalidates the result.

## Patch boundary

Only [`patches/patch_modelopt_w4a16_native_w4a4.py`](patches/patch_modelopt_w4a16_native_w4a4.py) is allowed. It must remain narrow and idempotent. The 40 aggregate routed-expert targets stay on B12X; all 121 ordinary targets require published finite-positive input scales before native W4A4 routing.

## Canonical evidence

- Run: `benchmarks/runs/qwen36-v025-mtp-20260714`
- GSM8K: 86.50% ± 0.94%, 1,319/1,319
- MTP acceptance: 78.56%
- Output throughput: c1 93.05, c8 287.18, c32 397.98 tok/s
- Image: `sha256:37b90ed38c415e1846aef8ffaaca8c3d39ec58bd68221eaa722a3d1e5e0387f1`

Do not hand-edit metrics. Generate docs from machine-readable evidence, then verify and regenerate the manifest.

## Validation order

1. Confirm the explicitly released GB10 host and avoid unrelated services/nodes.
2. Build the digest-pinned `Dockerfile.v025` image.
3. Run `audit_runtime_v025.py` and the 121-scale audit.
4. Inspect complete startup logs for required and forbidden markers.
5. Run deterministic semantic and long-generation gates.
6. Reuse complete equivalent evaluations only through `import_mtp_release_evidence.py`; require source/destination hashes.
7. On a repackaged clean image, run `run_mtp_equivalence_gate.sh`, not another full evaluation.
8. Generate HTML/docs, verify release JSON, check the manifest, and run both public scanners.
9. Stop the exact Qwen container and confirm the GB10 is released.

## Public hygiene

Commit only curated summaries, manifests, documentation, source, and tests. Never commit raw lm-eval samples, logs, telemetry streams, private IPs/paths, credentials, PID files, model weights, or caches. Public metrics must contain no unrelated experimental runtime names or labels.

## Completion

A release is complete only when tests pass, JSON/docs/HTML agree, checksums verify, the staged tree passes both scanners, a clean checkout reproduces command resolution, the exact container is stopped, and remote `main` plus Pages match the verified commit.
