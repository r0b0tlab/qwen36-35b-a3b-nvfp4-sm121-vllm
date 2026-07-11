# NVIDIA Qwen3.6-35B-A3B-NVFP4 model-card and runtime review

Final review: 2026-07-11

## Checkpoint identity

- Hub repository: `nvidia/Qwen3.6-35B-A3B-NVFP4`
- Pinned revision: `491c2f1ea524c639598bf8fa787a93fed5a6fbce`
- Architecture: `Qwen3_5MoeForConditionalGeneration`
- Model type: `qwen3_5_moe`
- Indexed tensor keys: 124,468
- Indexed tensor bytes: 23,407,580,856
- Safetensors shards: 3
- MTP keys: 19
- Visual keys: 333

The published checkpoint is already quantized. This repository audits and serves it read-only; it does not recalibrate, reconvert, or redistribute weights.

## Quantization interpretation

The artifact uses NVIDIA ModelOpt mixed FP8/NVFP4 metadata and requires `modelopt_mixed` in this vLLM lineage.

The checkpoint labels 161 targets `W4A16_NVFP4`. Forty aggregate routed-expert targets remain on the native B12X path. The remaining 121 ordinary linear targets—120 shared-expert projections plus `lm_head`—all carry complete finite-positive published `input_scale` tensors. The validated image-local patch consumes those scales through native W4A4 ModelOpt linear execution rather than the stock weight-only Marlin route.

Checkpoint tensor files and metadata remain unchanged.

## Accepted runtime

- NVIDIA GB10 / SM121;
- vLLM `0.24.1.dev0+gee0da84ab.d20260702`;
- CUDA 13.0 and PyTorch 2.11 cu130;
- `modelopt_mixed`;
- FlashInfer attention;
- native `FlashInferFP8ScaledMMLinearKernel`;
- native `FlashInferCutlassNvFp4LinearKernel`;
- native `FLASHINFER_B12X` routed-expert MoE;
- FP8 KV cache;
- MTP K=2 with Triton draft MoE;
- effective PIECEWISE CUDA graphs.

Marlin, weight-only W4A16 execution, emulation, metadata-only quantization, and fallback paths are rejected.

## KV-cache decision

FP8 KV is final for this release. NVFP4 model weights and NVFP4 KV cache are separate capabilities. NVFP4 KV was not adopted on SM121 because its scale-write and semantic-quality gates remain blocked; loader/prefill readiness is insufficient.

## Measured release

Canonical run: `nvidia-qwen36-native-20260711T100947Z`.

- GSM8K 0-shot flexible extract: 85.52% ± 0.97%, 1,319 samples;
- c1/c32 output throughput: 90.86 / 463.67 tok/s;
- final MTP acceptance: 85.63%;
- no failed serving benchmark requests;
- public report, artifact-consistency gate, SHA-256 manifest, and safety scan passed.

## Sources

- NVIDIA model card: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
- vLLM: https://github.com/vllm-project/vllm
- NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- Qwen: https://github.com/QwenLM/Qwen3