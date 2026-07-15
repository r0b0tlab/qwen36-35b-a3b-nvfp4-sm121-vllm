# NVIDIA Qwen3.6 35B-A3B NVFP4 on GB10

![NVIDIA Qwen3.6 35B-A3B NVFP4 on one GB10](assets/qwen36-gb10-hero.svg)

[![GPU: GB10 / SM121](https://img.shields.io/badge/GPU-GB10%20%2F%20SM121-76B900)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![vLLM: 0.25.0](https://img.shields.io/badge/vLLM-0.25.0-5B8DEF)](https://github.com/vllm-project/vllm)
[![Model: NVIDIA Qwen3.6 35B-A3B](https://img.shields.io/badge/Model-NVIDIA%20Qwen3.6%2035B--A3B-62F6FF)](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)

A correctness-gated, native SM121 deployment for NVIDIA's published [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) checkpoint on one NVIDIA GB10. The checkpoint is mounted read-only; this repository does not recalibrate, rewrite, or redistribute model weights.

## Release status

**Complete.** vLLM 0.25.0 runtime, native-kernel, published-scale, semantic, long-generation, MTP, evidence-integrity, clean-image equivalence, and full 262,144-token context gates passed.

```text
Run ID:           qwen36-v025-mtp-20260714
Model revision:   491c2f1ea524c639598bf8fa787a93fed5a6fbce
Image ID:         sha256:37b90ed38c415e1846aef8ffaaca8c3d39ec58bd68221eaa722a3d1e5e0387f1
Base digest:      sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5
HTML SHA-256:     3d54610f2b061a549938a230e3e075ef8b47f3ac3f7a89ef37f4601ac3c742d9
Manifest SHA-256: 7a5c49c1c6a8b219b8a22d9ecf1d750b872b48c464169e214f49fd4871658172
```

- [Self-contained report](docs/index.html)
- [Machine-readable evidence](benchmarks/runs/qwen36-v025-mtp-20260714/)
- [Published report](https://r0b0tlab.github.io/qwen36-35b-a3b-nvfp4-sm121-vllm/)

## Results

| Gate | Result |
|---|---:|
| GSM8K 0-shot flexible extract | **86.50% ± 0.94%** |
| GSM8K samples | **1,319 / 1,319** |
| MTP acceptance | **78.56%** |
| Maximum validated context, base AR and MTP K=2 | **262,144 tokens** |
| Published input scales | **121 / 121** |
| Clean-image semantic and long-generation gates | **Passed** |

GSM8K used local chat completions, the checkpoint chat template, 0-shot greedy decoding, `enable_thinking=false`, 2,048 generation tokens, and zero retries.

### Selected concurrency points

Random 2,048-token inputs and exact 512-token outputs. Three repetitions per point; repetition 1 is warm-up and repetitions 2–3 are averaged.

| Concurrency | Output tok/s | Mean TTFT | P99 TTFT |
|---:|---:|---:|---:|
| 1 | 93.05 | 357.47 ms | 382.45 ms |
| 8 | 287.18 | 1353.20 ms | 2141.14 ms |
| 32 | 397.98 | 10142.40 ms | 18016.79 ms |

### Full architectural context

A separate c1 context profile used an explicit 6 GiB FP8 KV allocation instead of the c32 throughput profile's aggregate KV allocation. Both base AR and production MTP K=2 passed live-tokenized controls at 65,536, 131,072, and 196,608 tokens, then passed the 262,144-token ceiling.

| Context gate | Base AR | MTP K=2 |
|---|---:|---:|
| Measured KV capacity | **609,637** | **528,230** |
| Near-window prompt range | 261,368–261,373 | 261,368–261,373 |
| Retrieval positions | begin / quarter / middle / three-quarter / end | begin / quarter / middle / three-quarter / end |
| Ordered dual-code retrieval | Passed | Passed |
| Forced output after 261,371-token prompt | 512 tokens | 512 tokens |
| Minimum host memory available | 51.12 GiB | 46.47 GiB |
| Added swap | 0 MiB | 0 MiB |
| Long-context MTP acceptance | — | **75.64%** |

The largest prompt-plus-output contract was 261,883 tokens, leaving 261 tokens of margin. This supports functional near-window retrieval and bounded generation—not a general reasoning-quality claim across arbitrary 262K prompts.

### Clean-image equivalence canaries

The clean image passed deterministic semantics, long generation, positive c1/c32 functional canaries, native-marker checks, and the exact runtime audit. Canary timing is deliberately excluded from headline performance because it is a packaging gate rather than the three-repetition benchmark methodology.

## Accepted runtime

| Component | Value |
|---|---|
| Base | vLLM 0.25.0 commit `702f4814fe54fabff350d43cb753ae3e47c0c276` |
| Torch / CUDA | `2.11.0+cu130` / `13.0` |
| FlashInfer | `0.6.13` |
| CUTLASS DSL / NCCL | `4.5.2` / `2.28.9` |
| Quantization | `modelopt_mixed` |
| KV cache | FP8 |
| Attention | FlashInfer |
| Linear kernels | FlashInfer FP8 + CUTLASS NVFP4 |
| Routed-expert MoE | `FLASHINFER_B12X` |
| Speculative decoding | MTP K=2, Triton draft MoE |
| CUDA graphs | Effective `PIECEWISE` |
| Throughput profile limits | 65,536 model length; 32 sequences; 32,768 batched tokens |
| Maximum-context profile | 262,144 model length; 1 sequence; 32,768 batched tokens; 6 GiB explicit FP8 KV |

The checkpoint labels 161 targets `W4A16_NVFP4`: 40 aggregate routed-expert targets remain B12X, while 121 ordinary linear targets are routed to native W4A4 only after all 121 published finite-positive `input_scale` tensors pass audit. The narrow idempotent patch is [`patches/patch_modelopt_w4a16_native_w4a4.py`](patches/patch_modelopt_w4a16_native_w4a4.py).

## Reproduce

### Build

```bash
docker build --pull -f Dockerfile.v025 \
  -t qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp .
```

The `FROM` image is pinned to `sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5`.

### Audit the image

```bash
docker run --rm --gpus all \
  qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp audit
```

### Launch

```bash
MODEL_HOST="$HOME/models/llm/nvfp4/nvidia/Qwen3.6-35B-A3B-NVFP4" \
  IMAGE=qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp \
  bash scripts/launch.sh
curl -fsS http://127.0.0.1:18080/health
curl -fsS http://127.0.0.1:18080/v1/models | python3 -m json.tool
```

### Launch the 262K c1 context profile

```bash
CONTAINER=qwen36-maxctx-mtp2-262144 \
  bash scripts/launch_max_context.sh mtp 262144
until curl -fsS http://127.0.0.1:18080/health >/dev/null; do
  docker ps --quiet --filter name=qwen36-maxctx-mtp2-262144 | grep -q . || exit 1
  sleep 5
done
python3 scripts/run_max_context_gate.py \
  --profile mtp2 --max-depth 262144 \
  --output /tmp/qwen36-max-context.json
docker stop qwen36-maxctx-mtp2-262144
docker rm qwen36-maxctx-mtp2-262144
```

### Verify evidence

```bash
python3 scripts/verify_release.py benchmarks/runs/qwen36-v025-mtp-20260714 docs/index.html
(cd benchmarks/runs/qwen36-v025-mtp-20260714 && sha256sum -c MANIFEST.sha256)
python3 scripts/public_safety_scan.py
python3 scripts/public_tree_guard.py
python3 -m unittest discover -s tests -v
```

## Evidence policy

The full evaluation is not repeated when model, runtime, serving flags, and methodology are equivalent. [`evidence-reuse.json`](benchmarks/runs/qwen36-v025-mtp-20260714/evidence-reuse.json) records SHA-256 hashes for imported measurements. The clean image independently passes runtime/native audits, semantics, long generation, and short c1/c32 canaries. Raw samples, private paths, logs, and telemetry streams are excluded.

## Limitations

- One GB10 / SM121; results are not multi-node claims.
- FP8 KV is the validated baseline; NVFP4 KV is not adopted.
- Selected c1/c8/c32 points are reported; absent optional suites are omitted.
- The 262K evidence is a c1 functional/retrieval qualification, not a concurrency or broad long-context reasoning benchmark.
- Model weights remain governed by NVIDIA's upstream terms and are not redistributed here.

## Credits

- [NVIDIA](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) for the calibrated ModelOpt checkpoint.
- [vLLM](https://github.com/vllm-project/vllm) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for the serving and kernel stacks.
- Repository code is MIT licensed; see [LICENSE](LICENSE).
