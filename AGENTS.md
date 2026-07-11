# AGENTS.md

## Repository purpose

This repository reproduces and benchmarks NVIDIA's pinned `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint on one NVIDIA GB10 / SM121. Correctness, effective kernel selection, provenance, and artifact hygiene are release gates—not caveats.

## Immutable model identity

```text
Hub repository: nvidia/Qwen3.6-35B-A3B-NVFP4
Revision:       491c2f1ea524c639598bf8fa787a93fed5a6fbce
Architecture:   Qwen3_5MoeForConditionalGeneration
Format:         NVIDIA ModelOpt mixed FP8/NVFP4
```

Never recalibrate, re-quantize, rewrite, or commit the published checkpoint. Mount it read-only.

## Final accepted runtime

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

Effective native markers:

- `FlashInferFP8ScaledMMLinearKernel` for FP8 linear targets;
- `R0B0TLAB_NATIVE_W4A4_FROM_W4A16` for the checkpoint-specific ordinary-linear reroute;
- `FlashInferCutlassNvFp4LinearKernel` for native NVFP4 W4A4 GEMM;
- `FLASHINFER_B12X` for routed-expert NVFP4 MoE;
- FlashInfer attention;
- MTP K=2 with Triton draft MoE;
- effective `PIECEWISE` CUDA graphs.

Any Marlin, weight-only W4A16 execution, emulation, missing-scale path, or fallback marker invalidates a performance result.

## Runtime patch boundary

The pinned artifact labels 161 targets `W4A16_NVFP4`:

- 40 aggregate routed-expert targets remain on native B12X;
- 120 shared-expert projections plus `lm_head` are ordinary linear targets.

All 121 ordinary targets must have present, finite, positive published `input_scale` tensors before rerouting. The only accepted implementation is [`patches/patch_modelopt_w4a16_native_w4a4.py`](patches/patch_modelopt_w4a16_native_w4a4.py), covered by [`scripts/audit_w4a16_input_scales.py`](scripts/audit_w4a16_input_scales.py). Do not broaden the patch, synthesize scales, or modify model tensors/metadata.

If the base image or vLLM lineage changes, re-audit the dispatcher source, rebuild, recapture startup markers, rerun semantics, and repeat the full campaign. Never assume the patch still applies cleanly.

## KV-cache decision

FP8 KV is final for this release. NVFP4 KV is **not** production-ready on SM121 because its separate scale-write and semantic-quality gates remain blocked. Do not advertise NVFP4 KV based on loader readiness, prefill success, or capacity alone.

## Canonical release evidence

```text
Run ID:       nvidia-qwen36-native-20260711T100947Z
Run root:     benchmarks/runs/nvidia-qwen36-native-20260711T100947Z/
HTML report:  docs/index.html
Image digest: sha256:a072794deee4b7875f4cbc40fb189a674188d7f7a10f32bf45038527c9485cfa
Repository:   https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-sm121-vllm
Pages:        https://r0b0tlab.github.io/qwen36-35b-a3b-nvfp4-sm121-vllm/
```

Accepted headline values:

- GSM8K 0-shot flexible extract: 85.52% ± 0.97%, 1,319/1,319 samples;
- output throughput: c1 90.86 tok/s, c32 463.67 tok/s;
- c32 efficiency: 11.36 output tokens/J;
- final MTP acceptance: 85.63%;
- no failed concurrency requests;
- llama-benchy coherence passed at four context depths.

Do not update these values manually. Regenerate the report from machine-readable evidence, then run the release verifier and manifest writer.

## Required validation order

1. Confirm the host is the explicitly released GB10 node—not the protected Hy3 node.
2. Run `scripts/audit_runtime.py` and record the exact image digest/runtime versions.
3. Run `scripts/audit_w4a16_input_scales.py` against the pinned read-only checkpoint.
4. Inspect the complete startup log independently for requested and effective linear, MoE, attention, KV, MTP, and CUDA-graph modes.
5. Reject any Marlin/W4A16-weight-only/emulation/fallback marker.
6. Run `scripts/verify_server.py`, `scripts/run_semantic_gate.py`, and `scripts/run_long_generation.py`.
7. Run GSM8K with `local-chat-completions`, `--apply_chat_template`, 0-shot greedy decoding, 2,048 generation tokens, and `chat_template_kwargs={"enable_thinking": false}`.
8. Run c1–c32 concurrency, llama-benchy, telemetry, and energy only after semantics pass.
9. Generate the HTML report from measured JSON.
10. Run `scripts/verify_release.py`, `scripts/write_manifest.py`, and `scripts/public_safety_scan.py`.

Readiness, HTTP 200, non-empty output, native MoE selection, or process exit zero alone do not establish valid logits or native ordinary-linear routing.

## Reproduction command

After the validated server is healthy:

```bash
RUN_ID="nvidia-qwen36-native-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/run_full_campaign.sh "$RUN_ID"
```

Use a persistent Herdr workspace for the full campaign. The script is fail-fast and performs no upload.

## Operational isolation

- NVIDIA Qwen work runs only on an explicitly released GB10 node.
- The separate Hy3 calibration node is protected: read-only status checks only.
- Never stop, prune, clean, reconfigure, or consume storage/compute on the Hy3 node for this project.
- Never use broad `pkill`, wildcard deletion, Docker prune, or indiscriminate container cleanup.
- Stop only the exact completed Qwen container after preserving and verifying release evidence.

## Git and publication hygiene

Never commit:

- model weights or caches;
- `.venv-bench`;
- raw lm-eval samples/results;
- telemetry streams;
- raw server, campaign, or per-repetition logs;
- PID files, temporary canaries, compiler caches, or Python bytecode;
- credentials, tokens, private IP addresses, or absolute operator home paths.

Commit curated JSON summaries, runtime evidence, scale audits, scripts, docs, report HTML, and SHA-256 manifests. Run `scripts/public_safety_scan.py` against the entire tree before every commit intended for publication.

The repository is model-specific. Do not introduce benchmark claims, branding, or runtime assumptions from unrelated models. Credit NVIDIA, Qwen, ModelOpt, vLLM, FlashInfer, CUTLASS, lm-eval, and llama-benchy appropriately.

## Completion rule

This project is complete only when:

- repository code and documentation agree with the measured accepted profile;
- all release and safety gates pass;
- the clean Git tree contains no ignored/private artifacts;
- the completed serving container is stopped;
- the protected Hy3 workload remains active and unchanged;
- no external upload occurs without explicit authorization.