# Engine lineage

## Target

`nvidia/Qwen3.6-35B-A3B-NVFP4` at revision `491c2f1ea524c639598bf8fa787a93fed5a6fbce`.

## Validated image

```text
Image ID:    sha256:a072794deee4b7875f4cbc40fb189a674188d7f7a10f32bf45038527c9485cfa
vLLM:       0.24.1.dev0+gee0da84ab.d20260702
PyTorch:    2.11.0+cu130
CUDA:       13.0
FlashInfer: 0.6.12
GPU:        NVIDIA GB10 / SM121
```

## Effective runtime

- ModelOpt mixed loader: `modelopt_mixed`;
- FP8 linear: `FlashInferFP8ScaledMMLinearKernel`;
- 121 calibrated W4A16-labelled ordinary targets: checkpoint-specific native W4A4 reroute;
- NVFP4 linear: `FlashInferCutlassNvFp4LinearKernel`;
- routed-expert MoE: `FLASHINFER_B12X`;
- attention: FlashInfer;
- KV cache: FP8;
- speculative decode: MTP K=2 with Triton draft MoE;
- CUDA graph mode: PIECEWISE.

The exact image digest, versions, backend markers, launch profile, MTP counters, and KV capacity are preserved in the canonical run's `runtime-manifest.json`.

## Native-path policy

Requested and effective behavior are independent gates. HTTP readiness, successful generation, or native MoE selection does not prove ordinary-linear routing. The complete startup log must contain the native reroute and linear markers and no selected Marlin, weight-only W4A16, or emulation path.

The runtime patch is local and version-sensitive. Re-audit it whenever vLLM, the base image, or the checkpoint revision changes.

## KV-cache decision

FP8 KV is the final validated baseline. NVFP4 KV is not adopted on SM121 because the separate scale-write and semantic-quality gates remain blocked.

## Protected infrastructure

The benchmark ran only on the released Qwen GB10. The separate active Hy3 calibration node was checked read-only and was never used for deployment, benchmarking, build, or cleanup.