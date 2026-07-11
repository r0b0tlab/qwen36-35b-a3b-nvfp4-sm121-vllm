# Native W4A4 linear reroute on GB10 / SM121

## Why this patch exists

NVIDIA's pinned `Qwen3.6-35B-A3B-NVFP4` artifact has two relevant descriptions:

- `hf_quant_config.json` labels 161 targets `W4A16_NVFP4`;
- `config.json` declares 4-bit input activations for the same mixed-precision group, and the checkpoint stores calibrated `input_scale` tensors.

The 161 targets split into:

- 40 aggregate routed-expert targets, executed by the native B12X MoE path using per-expert scales;
- 120 shared-expert linear projections plus `lm_head`, for 121 ordinary linear targets.

The audited 121 linear targets all contain `weight`, `weight_scale`, `weight_scale_2`, and finite positive `input_scale` tensors. At revision `491c2f1ea524c639598bf8fa787a93fed5a6fbce`, observed input scales span `0.0041155135` to `0.0654761940`.

## Upstream behavior

The vLLM lineage in the base image maps `W4A16_NVFP4` ordinary linear layers to `ModelOptNvFp4W4A16LinearMethod`. That class discards `input_scale` and directly instantiates the Marlin adapter. Setting `--linear-backend flashinfer_cutlass` does not override this hard pin.

This produced a hard release failure:

```text
Your GPU does not have native support for FP4 computation ... leveraging the Marlin kernel.
```

The rejected run is retained under `benchmarks/runs/fp8-baseline/` and must not feed release claims.

## Dedicated-image patch

`patches/patch_modelopt_w4a16_native_w4a4.py` changes only the ordinary-linear routing branch:

```text
W4A16_NVFP4 LinearBase -> ModelOptNvFp4LinearMethod
```

That method consumes the published `input_scale` tensors and selects the native SM121 kernel. `RoutedExperts` are untouched and remain on the independently selected B12X MoE backend. Model weights and checkpoint metadata remain read-only and unchanged.

The patch emits an auditable startup marker:

```text
R0B0TLAB_NATIVE_W4A4_FROM_W4A16
```

## Validated image and kernel contract

```text
Image:        sha256:a072794deee4b7875f4cbc40fb189a674188d7f7a10f32bf45038527c9485cfa
FP8 linear:   FlashInferFP8ScaledMMLinearKernel
NVFP4 linear: FlashInferCutlassNvFp4LinearKernel
NVFP4 MoE:    FLASHINFER_B12X
KV cache:     FP8
```

The complete server log contains no Marlin or emulation marker. `scripts/capture_release.py` enforces this as a hard failure.

## Validation commands

```bash
docker run --rm -i --entrypoint python3 \
  -v "$MODEL_DIR:/models/model:ro" \
  -v "$PWD:/workspace:ro" \
  qwen36-35b-a3b-nvfp4-sm121-vllm:native-w4a4-v1 \
  /workspace/scripts/audit_w4a16_input_scales.py /models/model

python3 scripts/run_semantic_gate.py \
  --base-url http://127.0.0.1:18080 \
  --output benchmarks/runs/fp8-native-w4a4-v1/semantic_gate/results.json

python3 scripts/run_long_generation.py \
  --base-url http://127.0.0.1:18080 \
  --output benchmarks/runs/fp8-native-w4a4-v1/long_generation.json
```

## Scope

This establishes the native FP8-KV correctness baseline. It does not make NVFP4-KV functional on SM121 and does not modify or supersede the separate Hy3 workflow.
