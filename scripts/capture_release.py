#!/usr/bin/env python3
"""Capture sanitized vLLM v0.25 MTP runtime and native-path evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

BASE_IMAGE_DIGEST = "sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5"
MARKER_ALTERNATIVES = {
    "modelopt_mixed": ("modelopt_mixed", "ModelOpt mixed", "ModelOpt NVFP4"),
    "fp8_dense": ("FlashInferFP8ScaledMMLinearKernel",),
    "nvfp4_dense": ("FlashInferCutlassNvFp4LinearKernel",),
    "w4a16_native_reroute": ("R0B0TLAB_NATIVE_W4A4_FROM_W4A16",),
    "target_moe_native": ("Using 'FLASHINFER_B12X' NvFp4 MoE backend", "FLASHINFER_B12X"),
    "flashinfer_attention": ("FLASHINFER attention backend", "FlashInfer attention", "attention_backend=flashinfer"),
}


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read().decode()


def metric_value(text: str, metric: str) -> float | None:
    values = []
    for line in text.splitlines():
        if line.startswith(metric + "{") or line.startswith(metric + " "):
            try:
                values.append(float(line.rsplit(" ", 1)[1]))
            except (ValueError, IndexError):
                pass
    return sum(values) if values else None


def optional_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--container", default=os.getenv("CONTAINER", "qwen36-nvfp4-vllm-v025-mtp"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "18080")))
    parser.add_argument("--model-revision", default="491c2f1ea524c639598bf8fa787a93fed5a6fbce")
    args = parser.parse_args()
    run = Path(args.run_dir)
    evidence = run / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)

    inspect = json.loads(command("docker", "inspect", args.container))[0]
    image = json.loads(command("docker", "image", "inspect", inspect["Image"]))[0]
    image_labels = image.get("Config", {}).get("Labels") or {}
    env = {}
    for item in inspect["Config"].get("Env", []):
        key, _, value = item.partition("=")
        env[key] = value

    models = json.loads(fetch(f"http://127.0.0.1:{args.port}/v1/models"))
    metrics = fetch(f"http://127.0.0.1:{args.port}/metrics")
    logs = command("docker", "logs", args.container)
    forbidden = re.compile(
        r"Marlin kernel|marlin\.py|Using 'MARLIN'|W4A16.*weight.only|falling back.*emulation|Using 'EMULATION'|missing.*input.scale",
        re.IGNORECASE,
    )
    match = forbidden.search(logs)
    if match:
        raise SystemExit(f"forbidden runtime marker found: {match.group(0)}")

    audit_path = run / "evidence" / "runtime-audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        packages = audit.get("packages", {})
        versions = {
            "torch": audit.get("torch"),
            "cuda": audit.get("torch_cuda"),
            "vllm": packages.get("vllm") or audit.get("vllm_api_version"),
            "flashinfer": packages.get("flashinfer-python"),
        }
    else:
        versions = json.loads(
            command(
                "docker", "exec", args.container, "python3", "-c",
                "import json,torch,vllm,flashinfer; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'vllm':vllm.__version__,'flashinfer':getattr(flashinfer,'__version__','unknown')}))",
            ).strip().splitlines()[-1]
        )
    gpu = command(
        "nvidia-smi", "--query-gpu=name,compute_cap,power.limit", "--format=csv,noheader,nounits"
    ).strip().splitlines()[0].split(",")

    marker_status = {
        name: any(option in logs for option in alternatives)
        for name, alternatives in MARKER_ALTERNATIVES.items()
    }
    marker_status["effective_piecewise"] = "PIECEWISE" in logs
    marker_status["mtp_configured"] = (
        "SpeculativeConfig(method='mtp'" in logs
        or '"method":"mtp"' in logs
        or "method='mtp'" in logs
    )
    missing = [name for name, present in marker_status.items() if not present]
    if missing:
        raise SystemExit(f"missing required native markers: {', '.join(missing)}")

    expected_env = {
        "KV_CACHE_DTYPE": "fp8",
        "QUANTIZATION": "modelopt_mixed",
        "MOE_BACKEND": "flashinfer_b12x",
        "LINEAR_BACKEND": "flashinfer_cutlass",
        "ATTENTION_BACKEND": "flashinfer",
    }
    for key, expected in expected_env.items():
        if env.get(key) != expected:
            raise SystemExit(f"{key}: expected {expected!r}, got {env.get(key)!r}")
    spec = json.loads(env.get("SPECULATIVE_CONFIG", "{}"))
    if spec != {"method": "mtp", "num_speculative_tokens": 2, "moe_backend": "triton"}:
        raise SystemExit(f"unexpected MTP config: {spec}")

    draft = metric_value(metrics, "vllm:spec_decode_num_draft_tokens_total")
    accepted = metric_value(metrics, "vllm:spec_decode_num_accepted_tokens_total")
    if not draft or accepted is None or accepted <= 0 or accepted > draft:
        raise SystemExit(f"invalid MTP counters: accepted={accepted} draft={draft}")
    acceptance = accepted / draft
    kv_match = re.search(r"GPU KV cache size:\s*([0-9,]+) tokens", logs)
    moe_match = re.search(r"Using '([^']+)' NvFp4 MoE backend", logs)

    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "served_model": models["data"][0]["id"],
        "model_revision": args.model_revision,
        "max_model_len": models["data"][0].get("max_model_len"),
        "hardware": {
            "gpu": gpu[0].strip(),
            "compute_capability": gpu[1].strip(),
            "power_limit_w": optional_float(gpu[2]),
            "architecture": "aarch64",
        },
        "runtime": versions,
        "image_id": inspect["Image"],
        "image_repo_digests": image.get("RepoDigests") or [],
        "image_labels": image_labels,
        "base_image_digest": BASE_IMAGE_DIGEST,
        "profile": {key: env.get(key) for key in (
            "KV_CACHE_DTYPE", "QUANTIZATION", "MOE_BACKEND", "LINEAR_BACKEND",
            "SPECULATIVE_CONFIG", "MAX_MODEL_LEN", "MAX_NUM_SEQS",
            "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION",
            "ATTENTION_BACKEND", "LANGUAGE_MODEL_ONLY",
        )},
        "effective_cuda_graph_mode": "PIECEWISE",
        "effective_moe_backend": moe_match.group(1) if moe_match else None,
        "kv_cache_tokens": int(kv_match.group(1).replace(",", "")) if kv_match else None,
        "mtp_draft_tokens_total": draft,
        "mtp_accepted_tokens_total": accepted,
        "mtp_acceptance_rate": acceptance,
        "native_marker_gate": marker_status,
        "semantic_profile": "FP8 KV final; NVFP4 KV not adopted on SM121",
    }
    (run / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (evidence / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    selected = [line for line in logs.splitlines() if (
        any(option in line for alternatives in MARKER_ALTERNATIVES.values() for option in alternatives)
        or "SpeculativeConfig(method='mtp'" in line
        or "kv_cache_dtype=fp8" in line
        or "PIECEWISE" in line
    )]
    (evidence / "native-marker-log-lines.txt").write_text("\n".join(selected) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
