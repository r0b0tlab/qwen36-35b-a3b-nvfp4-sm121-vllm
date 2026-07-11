#!/usr/bin/env python3
"""Capture sanitized runtime and native-path evidence for the NVIDIA Qwen release."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

MARKER_ALTERNATIVES = {
    "modelopt_mixed": ("modelopt_mixed", "ModelOpt mixed", "ModelOpt NVFP4"),
    "fp8_dense": ("CutlassFP8ScaledMMLinearKernel", "FP8ScaledMMLinearKernel"),
    "nvfp4_dense": ("FlashInferCutlassNvFp4LinearKernel", "NvFp4LinearKernel"),
    "w4a16_native_reroute": ("R0B0TLAB_NATIVE_W4A4_FROM_W4A16",),
    "target_moe_native": (
        "Using 'FLASHINFER_B12X' NvFp4 MoE backend",
        "Using 'FLASHINFER_CUTEDSL' NvFp4 MoE backend",
        "Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend",
        "Using 'FLASHINFER_CUTEDSL_BATCHED' NvFp4 MoE backend",
    ),
    "flashinfer_attention": ("FLASHINFER attention backend", "FlashInfer attention"),
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
    parser.add_argument("--container", default=os.getenv("CONTAINER", "qwen36-nvfp4-vllm"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "18080")))
    parser.add_argument("--model-revision", default="491c2f1ea524c639598bf8fa787a93fed5a6fbce")
    args = parser.parse_args()

    run = Path(args.run_dir)
    evidence = run / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)

    inspect = json.loads(command("docker", "inspect", args.container))[0]
    env = {}
    for item in inspect["Config"].get("Env", []):
        key, _, value = item.partition("=")
        env[key] = value

    models = json.loads(fetch(f"http://127.0.0.1:{args.port}/v1/models"))
    metrics = fetch(f"http://127.0.0.1:{args.port}/metrics")
    logs = command("docker", "logs", args.container)
    if re.search(r"Marlin kernel|marlin\.py|Using 'MARLIN' NvFp4 MoE backend", logs, re.IGNORECASE):
        raise SystemExit("forbidden Marlin marker found in server logs")
    if re.search(r"falling back.*emulation|Using 'EMULATION' NvFp4 MoE backend", logs, re.IGNORECASE):
        raise SystemExit("forbidden emulation marker found in server logs")
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
    # CUDA graph mode and MTP are conditional evidence: record them, but do not
    # claim MTP acceptance unless counters are present and nonzero.
    marker_status["effective_piecewise"] = "setting cudagraph_mode=PIECEWISE" in logs or "PIECEWISE" in logs
    marker_status["mtp_configured"] = "SpeculativeConfig(method='mtp'" in logs or '"method":"mtp"' in logs
    missing = [name for name, present in marker_status.items() if not present and name != "mtp_configured"]
    if missing:
        raise SystemExit(f"missing required native markers: {', '.join(missing)}")
    if env.get("KV_CACHE_DTYPE") != "fp8":
        raise SystemExit(f"release baseline requires FP8 KV; got {env.get('KV_CACHE_DTYPE')!r}")
    if env.get("QUANTIZATION") not in {"modelopt_mixed", None}:
        raise SystemExit(f"unexpected quantization profile: {env.get('QUANTIZATION')!r}")

    draft = metric_value(metrics, "vllm:spec_decode_num_draft_tokens_total")
    accepted = metric_value(metrics, "vllm:spec_decode_num_accepted_tokens_total")
    acceptance = accepted / draft if draft else None
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
        "profile": {
            key: env.get(key)
            for key in (
                "KV_CACHE_DTYPE", "QUANTIZATION", "MOE_BACKEND", "LINEAR_BACKEND",
                "SPECULATIVE_CONFIG", "MAX_MODEL_LEN", "MAX_NUM_SEQS",
                "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION",
                "ATTENTION_BACKEND", "LANGUAGE_MODEL_ONLY",
            )
        },
        "effective_cuda_graph_mode": "PIECEWISE" if marker_status["effective_piecewise"] else "unconfirmed",
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
    (evidence / "native-markers.txt").write_text(
        "\n".join(option for alternatives in MARKER_ALTERNATIVES.values() for option in alternatives) + "\n"
    )
    selected = [
        line for line in logs.splitlines()
        if any(option in line for alternatives in MARKER_ALTERNATIVES.values() for option in alternatives)
        or "SpeculativeConfig(method='mtp'" in line
        or "kv_cache_dtype=fp8" in line
        or "PIECEWISE" in line
    ]
    (evidence / "native-marker-log-lines.txt").write_text("\n".join(selected) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
