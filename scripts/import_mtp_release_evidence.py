#!/usr/bin/env python3
"""Import completed vLLM v0.25 MTP evidence without rerunning equivalent work."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
VLLM_COMMIT = "702f4814fe54fabff350d43cb753ae3e47c0c276"
BASE_DIGEST = "sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5"
PROFILE = {
    "quantization": "modelopt_mixed",
    "kv_cache_dtype": "fp8",
    "attention_backend": "flashinfer",
    "moe_backend": "flashinfer_b12x",
    "linear_backend": "flashinfer_cutlass",
    "speculative": {"method": "mtp", "num_speculative_tokens": 2, "moe_backend": "triton"},
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing required evidence: {path.name}")
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def metric(data: dict, *names: str):
    for name in names:
        if name in data:
            return data[name]
    return None


def summarize_throughput(profile: Path) -> dict:
    rows = []
    fields = {
        "request_throughput_rps": ("request_throughput",),
        "output_throughput_tok_s": ("output_throughput",),
        "total_throughput_tok_s": ("total_token_throughput", "total_throughput"),
        "mean_ttft_ms": ("mean_ttft_ms",),
        "p99_ttft_ms": ("p99_ttft_ms",),
        "mean_tpot_ms": ("mean_tpot_ms",),
        "p99_tpot_ms": ("p99_tpot_ms",),
        "mean_itl_ms": ("mean_itl_ms",),
        "p99_itl_ms": ("p99_itl_ms",),
    }
    for concurrency in (1, 8, 32):
        reps = [load(profile / "throughput" / f"c{concurrency}-r{rep}.json") for rep in (1, 2, 3)]
        if any(int(metric(rep, "failed") or 0) for rep in reps):
            raise SystemExit(f"failed request in c{concurrency} evidence")
        row: dict[str, object] = {
            "concurrency": concurrency,
            "repetitions": 3,
            "warmup_rep_dropped": 1,
        }
        for output_name, input_names in fields.items():
            values = [metric(rep, *input_names) for rep in reps[1:]]
            values = [float(value) for value in values if value is not None]
            row[output_name] = statistics.mean(values) if values else None
        row["completed"] = sum(int(metric(rep, "completed") or 0) for rep in reps[1:])
        row["failed"] = 0
        rows.append(row)
    return {
        "method": "vllm bench serve random 2048 input / 512 output / ignore_eos; three reps, first dropped",
        "scope": "selected MTP qualification points",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run")
    parser.add_argument("destination_run")
    args = parser.parse_args()
    source = Path(args.source_run).resolve()
    destination = Path(args.destination_run).resolve()
    profile = source / "profiles" / "mtp-k2"
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit("destination must be absent or empty")

    if (profile / "STATUS").read_text().strip() != "COMPLETE":
        raise SystemExit("MTP profile is not complete")
    summary = load(profile / "summary.json")
    spec = load(profile / "spec-metrics.json")
    gsm = load(source / "gsm8k" / "results.json")
    semantic = load(profile / "semantic.json")
    long_generation = load(profile / "long-generation.json")
    scales = load(profile / "evidence" / "w4a16-scales.json")

    if summary.get("profile") != "mtp-k2" or not summary.get("passed"):
        raise SystemExit("source profile gate failed")
    if not spec.get("passed") or not (0 < float(spec["accepted_tokens"]) <= float(spec["draft_tokens"])):
        raise SystemExit("invalid MTP metrics")
    if int(gsm.get("sample_count") or 0) != 1319:
        raise SystemExit("full GSM8K evidence must contain 1319 samples")
    if float(gsm.get("exact_match_flexible_extract") or 0) < 0.835:
        raise SystemExit("full GSM8K result is below the accepted floor")
    model_args = gsm.get("model_args") or {}
    if model_args.get("model") != "Qwen3.6-35B-A3B-NVFP4" or int(model_args.get("max_retries", -1)) != 0:
        raise SystemExit("GSM8K model or retry contract mismatch")
    if not semantic.get("passed") or not long_generation.get("passed") or not scales.get("passed"):
        raise SystemExit("semantic, long-generation, or scale gate failed")
    if int(scales.get("linear_input_scales_loaded") or 0) != 121:
        raise SystemExit("published input-scale count mismatch")

    concurrency = summarize_throughput(profile)
    outputs = {
        "gsm8k/results.json": gsm,
        "concurrency/summary.json": concurrency,
        "semantic_gate/results.json": semantic,
        "long_generation.json": long_generation,
        "spec-metrics.json": spec,
        "evidence/w4a16-input-scale-audit.json": {key: value for key, value in scales.items() if key != "model_path"},
    }
    source_files = {
        "gsm8k": source / "gsm8k" / "results.json",
        "profile_summary": profile / "summary.json",
        "spec_metrics": profile / "spec-metrics.json",
        "semantic": profile / "semantic.json",
        "long_generation": profile / "long-generation.json",
        "scale_audit": profile / "evidence" / "w4a16-scales.json",
    }
    for concurrency_level in (1, 8, 32):
        for rep in (1, 2, 3):
            source_files[f"c{concurrency_level}_r{rep}"] = profile / "throughput" / f"c{concurrency_level}-r{rep}.json"

    for relative, payload in outputs.items():
        write_json(destination / relative, payload)
    manifest = {
        "schema_version": 1,
        "evidence_policy": "hash-verified reuse; no duplicate full evaluation",
        "source_type": "completed-vllm-v025-mtp-qualification",
        "model_revision": MODEL_REVISION,
        "vllm_release_commit": VLLM_COMMIT,
        "base_image_digest": BASE_DIGEST,
        "profile": PROFILE,
        "source_sha256": {logical: sha(path) for logical, path in sorted(source_files.items())},
        "destination_sha256": {relative: sha(destination / relative) for relative in sorted(outputs)},
        "checks": {
            "full_gsm8k_1319": True,
            "zero_retries": True,
            "mtp_metrics_positive": True,
            "three_repetitions_first_dropped": True,
            "semantic_gate": True,
            "long_generation_gate": True,
            "published_scales_121": True,
        },
        "passed": True,
    }
    write_json(destination / "evidence-reuse.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
