# Closed-project handoff

The NVIDIA Qwen3.6-35B-A3B NVFP4 release workflow is complete. Do not rerun it unless validating a deliberate runtime/checkpoint change.

## Canonical identity

```text
Model:    nvidia/Qwen3.6-35B-A3B-NVFP4
Revision: 491c2f1ea524c639598bf8fa787a93fed5a6fbce
Run:      nvidia-qwen36-native-20260711T100947Z
Report:   docs/index.html
```

## Accepted profile

```text
modelopt_mixed
FP8 KV
FlashInfer attention
FlashInfer FP8 linear
checkpoint-specific native W4A4 ordinary-linear reroute
FlashInfer CUTLASS NVFP4 linear
FLASHINFER_B12X routed-expert MoE
MTP K=2 with Triton draft MoE
PIECEWISE CUDA graphs
```

The native patch is valid only for the pinned checkpoint after all 121 ordinary target scales pass `scripts/audit_w4a16_input_scales.py`. Any base-image/vLLM/checkpoint change requires rebuilding and repeating every correctness and benchmark gate.

## Final outcomes

- GSM8K 0-shot flexible extract: 85.52% ± 0.97%, 1,319/1,319;
- c1/c32 output throughput: 90.86 / 463.67 tok/s;
- MTP acceptance: 85.63%;
- four-depth llama-benchy coherence pass;
- release consistency, manifest, and public safety pass;
- FP8 KV retained; NVFP4 KV not adopted;
- no external upload performed during local closure.

## Operational boundary

The completed Qwen container should remain stopped unless explicitly needed. The independent Hy3 calibration node is protected and must receive read-only checks only from this project.

## If reopening

1. Read `README.md`, `AGENTS.md`, and `docs/NATIVE_W4A4_REROUTE.md`.
2. Verify the exact checkpoint revision and image digest.
3. Confirm an explicitly released GB10 node and read-only Hy3 isolation.
4. Run runtime and scale audits before launch.
5. Repeat semantics before any throughput claim.
6. Use a fresh run ID and do not overwrite canonical evidence.
7. Perform no external publication without explicit authorization.