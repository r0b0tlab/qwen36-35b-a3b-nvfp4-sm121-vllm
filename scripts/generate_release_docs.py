#!/usr/bin/env python3
"""Generate README and operator guidance from the canonical release evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODEL_REV = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
BASE_DIGEST = "sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    run = Path(args.run_dir).resolve()
    run_rel = run.relative_to(root)
    gsm = load(run / "gsm8k/results.json")
    conc = load(run / "concurrency/summary.json")
    spec = load(run / "spec-metrics.json")
    runtime = load(run / "runtime-manifest.json")
    equiv = load(run / "equivalence/summary.json")
    maxctx = load(run / "max-context.json")
    rows = conc["rows"]
    score = 100 * float(gsm["exact_match_flexible_extract"])
    stderr = 100 * float(gsm["exact_match_flexible_extract_stderr"])
    acceptance = 100 * float(spec["acceptance_rate"])
    context_acceptance = 100 * float(maxctx["mtp_k2"]["acceptance_rate"])
    html_hash = sha(root / "docs/index.html")
    manifest_hash = sha(run / "MANIFEST.sha256")
    image_id = runtime["image_id"]
    table = "\n".join(
        f"| {row['concurrency']} | {row['output_throughput_tok_s']:.2f} | {row['mean_ttft_ms']:.2f} ms | {row['p99_ttft_ms']:.2f} ms |"
        for row in rows
    )

    readme = f"""# NVIDIA Qwen3.6 35B-A3B NVFP4 on GB10

![NVIDIA Qwen3.6 35B-A3B NVFP4 on one GB10](assets/qwen36-gb10-hero.svg)

[![GPU: GB10 / SM121](https://img.shields.io/badge/GPU-GB10%20%2F%20SM121-76B900)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![vLLM: 0.25.0](https://img.shields.io/badge/vLLM-0.25.0-5B8DEF)](https://github.com/vllm-project/vllm)
[![Model: NVIDIA Qwen3.6 35B-A3B](https://img.shields.io/badge/Model-NVIDIA%20Qwen3.6%2035B--A3B-62F6FF)](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)

A correctness-gated, native SM121 deployment for NVIDIA's published [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) checkpoint on one NVIDIA GB10. The checkpoint is mounted read-only; this repository does not recalibrate, rewrite, or redistribute model weights.

## Release status

**Complete.** vLLM 0.25.0 runtime, native-kernel, published-scale, semantic, long-generation, MTP, evidence-integrity, clean-image equivalence, and full 262,144-token context gates passed.

```text
Run ID:           {run.name}
Model revision:   {MODEL_REV}
Image ID:         {image_id}
Base digest:      {BASE_DIGEST}
HTML SHA-256:     {html_hash}
Manifest SHA-256: {manifest_hash}
```

- [Self-contained report](docs/index.html)
- [Machine-readable evidence]({run_rel.as_posix()}/)
- [Published report](https://r0b0tlab.github.io/qwen36-35b-a3b-nvfp4-sm121-vllm/)

## Results

| Gate | Result |
|---|---:|
| GSM8K 0-shot flexible extract | **{score:.2f}% ± {stderr:.2f}%** |
| GSM8K samples | **1,319 / 1,319** |
| MTP acceptance | **{acceptance:.2f}%** |
| Maximum validated context, base AR and MTP K=2 | **262,144 tokens** |
| Published input scales | **121 / 121** |
| Clean-image semantic and long-generation gates | **Passed** |

GSM8K used local chat completions, the checkpoint chat template, 0-shot greedy decoding, `enable_thinking=false`, 2,048 generation tokens, and zero retries.

### Selected concurrency points

Random 2,048-token inputs and exact 512-token outputs. Three repetitions per point; repetition 1 is warm-up and repetitions 2–3 are averaged.

| Concurrency | Output tok/s | Mean TTFT | P99 TTFT |
|---:|---:|---:|---:|
{table}

### Full architectural context

A separate c1 context profile used an explicit 6 GiB FP8 KV allocation instead of the c32 throughput profile's aggregate KV allocation. Both base AR and production MTP K=2 passed live-tokenized controls at 65,536, 131,072, and 196,608 tokens, then passed the 262,144-token ceiling.

| Context gate | Base AR | MTP K=2 |
|---|---:|---:|
| Measured KV capacity | **{maxctx['base_ar']['measured_kv_capacity_tokens']:,}** | **{maxctx['mtp_k2']['measured_kv_capacity_tokens']:,}** |
| Near-window prompt range | 261,368–261,373 | 261,368–261,373 |
| Retrieval positions | begin / quarter / middle / three-quarter / end | begin / quarter / middle / three-quarter / end |
| Ordered dual-code retrieval | Passed | Passed |
| Forced output after 261,371-token prompt | 512 tokens | 512 tokens |
| Minimum host memory available | {maxctx['base_ar']['minimum_mem_available_gib']:.2f} GiB | {maxctx['mtp_k2']['minimum_mem_available_gib']:.2f} GiB |
| Added swap | 0 MiB | 0 MiB |
| Long-context MTP acceptance | — | **{context_acceptance:.2f}%** |

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
docker build --pull -f Dockerfile.v025 \\
  -t qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp .
```

The `FROM` image is pinned to `{BASE_DIGEST}`.

### Audit the image

```bash
docker run --rm --gpus all \\
  qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp audit
```

### Launch

```bash
MODEL_HOST="$HOME/models/llm/nvfp4/nvidia/Qwen3.6-35B-A3B-NVFP4" \\
  IMAGE=qwen36-35b-a3b-nvfp4-sm121-vllm:v025-mtp \\
  bash scripts/launch.sh
curl -fsS http://127.0.0.1:18080/health
curl -fsS http://127.0.0.1:18080/v1/models | python3 -m json.tool
```

### Launch the 262K c1 context profile

```bash
CONTAINER=qwen36-maxctx-mtp2-262144 \\
  bash scripts/launch_max_context.sh mtp 262144
until curl -fsS http://127.0.0.1:18080/health >/dev/null; do
  docker ps --quiet --filter name=qwen36-maxctx-mtp2-262144 | grep -q . || exit 1
  sleep 5
done
python3 scripts/run_max_context_gate.py \\
  --profile mtp2 --max-depth 262144 \\
  --output /tmp/qwen36-max-context.json
docker stop qwen36-maxctx-mtp2-262144
docker rm qwen36-maxctx-mtp2-262144
```

### Verify evidence

```bash
python3 scripts/verify_release.py {run_rel.as_posix()} docs/index.html
(cd {run_rel.as_posix()} && sha256sum -c MANIFEST.sha256)
python3 scripts/public_safety_scan.py
python3 scripts/public_tree_guard.py
python3 -m unittest discover -s tests -v
```

## Evidence policy

The full evaluation is not repeated when model, runtime, serving flags, and methodology are equivalent. [`evidence-reuse.json`]({run_rel.as_posix()}/evidence-reuse.json) records SHA-256 hashes for imported measurements. The clean image independently passes runtime/native audits, semantics, long generation, and short c1/c32 canaries. Raw samples, private paths, logs, and telemetry streams are excluded.

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
"""
    agents = f"""# AGENTS.md

## Purpose

Reproduce NVIDIA's pinned Qwen3.6 35B-A3B NVFP4 checkpoint on one GB10/SM121 with correctness, native-path, provenance, and public-artifact hygiene as release gates.

## Immutable identity

- Model: `nvidia/Qwen3.6-35B-A3B-NVFP4`
- Revision: `{MODEL_REV}`
- Mount weights read-only. Never recalibrate, rewrite, synthesize scales, or commit model/cache files.
- Base image digest: `{BASE_DIGEST}`
- vLLM commit: `702f4814fe54fabff350d43cb753ae3e47c0c276`

## Accepted profile

```text
QUANTIZATION=modelopt_mixed
KV_CACHE_DTYPE=fp8
ATTENTION_BACKEND=flashinfer
MOE_BACKEND=flashinfer_b12x
LINEAR_BACKEND=flashinfer_cutlass
SPECULATIVE_CONFIG={{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}}
GPU_MEMORY_UTILIZATION=0.88
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=32768
```

Separate maximum-context profile: `MAX_MODEL_LEN=262144`, `MAX_NUM_SEQS=1`, `MAX_NUM_BATCHED_TOKENS=32768`, and `--kv-cache-memory-bytes 6G`. Do not replace the throughput profile with this c1 profile or merge their performance claims.

Required effective markers: `FlashInferFP8ScaledMMLinearKernel`, `R0B0TLAB_NATIVE_W4A4_FROM_W4A16`, `FlashInferCutlassNvFp4LinearKernel`, `FLASHINFER_B12X`, FlashInfer attention, MTP K=2, and `PIECEWISE`. Any Marlin, weight-only W4A16 execution, emulation, missing-scale, or fallback marker invalidates the result.

## Patch boundary

Only [`patches/patch_modelopt_w4a16_native_w4a4.py`](patches/patch_modelopt_w4a16_native_w4a4.py) is allowed. It must remain narrow and idempotent. The 40 aggregate routed-expert targets stay on B12X; all 121 ordinary targets require published finite-positive input scales before native W4A4 routing.

## Canonical evidence

- Run: `{run_rel.as_posix()}`
- GSM8K: {score:.2f}% ± {stderr:.2f}%, 1,319/1,319
- MTP acceptance: {acceptance:.2f}%
- Output throughput: c1 {rows[0]['output_throughput_tok_s']:.2f}, c8 {rows[1]['output_throughput_tok_s']:.2f}, c32 {rows[2]['output_throughput_tok_s']:.2f} tok/s
- Full architectural context: 262,144 tokens passed for base AR and MTP K=2; long-context MTP acceptance {context_acceptance:.2f}%
- Image: `{image_id}`

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
"""
    bench = f"""# Benchmark evidence

Canonical release run: [`{run.name}`](runs/{run.name}/)

## Quality

- GSM8K test split: 1,319/1,319 samples.
- 0-shot local chat completions, chat template applied, thinking disabled, greedy decoding, 2,048-token budget, zero retries.
- Flexible-extract exact match: **{score:.2f}% ± {stderr:.2f}%**.

## Selected throughput

- Random 2,048-token inputs and exact 512-token outputs.
- c1, c8, c32; three repetitions each; first dropped, final two averaged.
- Output throughput: c1 **{rows[0]['output_throughput_tok_s']:.2f}**, c8 **{rows[1]['output_throughput_tok_s']:.2f}**, c32 **{rows[2]['output_throughput_tok_s']:.2f} tok/s**.
- MTP acceptance: **{acceptance:.2f}%**.

## Maximum context

- Separate c1 profile: 262,144 `max_model_len`, 6 GiB explicit FP8 KV, max sequences 1.
- Base AR KV capacity: **{maxctx['base_ar']['measured_kv_capacity_tokens']:,} tokens**.
- MTP K=2 KV capacity: **{maxctx['mtp_k2']['measured_kv_capacity_tokens']:,} tokens**.
- Five-position near-window retrieval, ordered dual-code retrieval, and 512 forced output tokens passed for both profiles.
- MTP K=2 accepted 534/706 drafted tokens (**{context_acceptance:.2f}%**) during the long-context gate.
- Claim boundary: functional retrieval and bounded generation, not arbitrary 262K reasoning quality.

## Reuse and equivalence

`evidence-reuse.json` hashes every imported source and curated destination artifact. `equivalence/summary.json` records clean-image runtime, semantic, long-generation, c1, and c32 regression gates. No duplicate full evaluation was run for the equivalent clean image.

## Integrity

```bash
cd {run_rel.as_posix()}
sha256sum -c MANIFEST.sha256
```
"""
    (root / "README.md").write_text(readme)
    (root / "AGENTS.md").write_text(agents)
    (root / "benchmarks/README.md").write_text(bench)
    print(json.dumps({"readme": str(root / "README.md"), "agents": str(root / "AGENTS.md"), "benchmarks": str(root / "benchmarks/README.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
