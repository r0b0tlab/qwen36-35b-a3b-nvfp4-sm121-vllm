#!/usr/bin/env python3
"""Consistency gate for the generated NVIDIA Qwen release artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"


def load(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError(f"missing: {path}")
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("html")
    args = parser.parse_args()
    run = Path(args.run_dir)
    html_path = Path(args.html)

    gsm = load(run / "gsm8k/results.json")
    concurrency = load(run / "concurrency/summary.json")
    llama = load(run / "llama-benchy/results.json")
    telemetry = load(run / "telemetry-summary.json")
    energy = load(run / "energy-efficiency.json")
    runtime = load(run / "runtime-manifest.json")
    page = html_path.read_text()

    assert runtime["model"] == MODEL
    assert runtime["model_revision"] == REVISION
    profile = runtime["profile"]
    assert profile["KV_CACHE_DTYPE"] == "fp8"
    assert profile["QUANTIZATION"] == "modelopt_mixed"
    assert profile["MOE_BACKEND"] == "flashinfer_b12x"
    assert profile["LINEAR_BACKEND"] == "flashinfer_cutlass"
    assert runtime["effective_cuda_graph_mode"] in {"PIECEWISE", "EAGER"}
    required_markers = {
        "modelopt_mixed", "fp8_dense", "nvfp4_dense", "w4a16_native_reroute",
        "target_moe_native", "flashinfer_attention",
    }
    assert all(runtime["native_marker_gate"].get(k) for k in required_markers)

    assert gsm["sample_count"] == 1319
    assert 0.0 <= gsm["exact_match_flexible_extract"] <= 1.0
    rows = concurrency["rows"]
    assert [x["concurrency"] for x in rows] == [1, 2, 4, 8, 16, 32]
    assert all(x["completed"] > 0 and x["failed"] == 0 for x in rows)
    assert len(llama.get("benchmarks", [])) >= 1
    assert telemetry.get("phases")
    assert len(energy.get("rows", [])) >= 1

    required = [
        "Qwen3.6 35B-A3B", "FP8", "ModelOpt", "FlashInfer",
        "Concurrency scaling", "Mean active power W", "NVFP4-KV candidate",
    ]
    for item in required:
        assert item in page, f"HTML missing {item!r}"
    forbidden = ["NVFP4 Fast", "MARLIN backend selected"]
    for item in forbidden:
        assert item.lower() not in page.lower(), f"HTML contains forbidden value {item!r}"
    assert page.count("<div") == page.count("</div>"), "unbalanced div elements"
    assert page.count("<section") == page.count("</section>"), "unbalanced section elements"
    assert "opacity:0" not in page

    print("PASS NVIDIA Qwen release artifact consistency gate")
    print(f"GSM8K={gsm['exact_match_flexible_extract']:.6f} samples={gsm['sample_count']}")
    print(f"concurrency_rows={len(rows)} llama_depths={len(llama['benchmarks'])}")
    print(f"html={html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
