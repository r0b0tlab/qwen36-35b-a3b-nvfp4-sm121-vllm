#!/usr/bin/env python3
"""Consistency gate for the generated NVIDIA Qwen vLLM v0.25 MTP release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
BASE_DIGEST = "sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5"


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
    audit = load(run / "evidence/runtime-audit.json")
    scales = load(run / "evidence/w4a16-input-scale-audit.json")
    semantic = load(run / "semantic_gate/results.json")
    long_generation = load(run / "long_generation.json")
    durability = load(run / "full/durability.summary.json")
    page = html_path.read_text()

    assert runtime["model"] == MODEL
    assert runtime["model_revision"] == REVISION
    assert runtime["base_image_digest"] == BASE_DIGEST
    assert runtime["runtime"]["vllm"] == "0.25.0"
    assert runtime["runtime"]["torch"] == "2.11.0+cu130"
    assert runtime["runtime"]["cuda"] == "13.0"
    assert runtime["runtime"]["flashinfer"] == "0.6.13"
    assert audit["status"] == "PASS" and not audit["failures"]
    assert scales["passed"] and scales["linear_input_scales_loaded"] == 121
    assert semantic["passed"] and long_generation["passed"]
    assert durability["passed"] and durability["completed"] == durability["requested"] == 250

    profile = runtime["profile"]
    assert profile["KV_CACHE_DTYPE"] == "fp8"
    assert profile["QUANTIZATION"] == "modelopt_mixed"
    assert profile["MOE_BACKEND"] == "flashinfer_b12x"
    assert profile["LINEAR_BACKEND"] == "flashinfer_cutlass"
    assert profile["ATTENTION_BACKEND"] == "flashinfer"
    assert json.loads(profile["SPECULATIVE_CONFIG"]) == {
        "method": "mtp", "num_speculative_tokens": 2, "moe_backend": "triton"
    }
    assert runtime["effective_cuda_graph_mode"] == "PIECEWISE"
    required_markers = {
        "modelopt_mixed", "fp8_dense", "nvfp4_dense", "w4a16_native_reroute",
        "target_moe_native", "flashinfer_attention", "effective_piecewise", "mtp_configured",
    }
    assert all(runtime["native_marker_gate"].get(key) for key in required_markers)
    assert 0 < runtime["mtp_accepted_tokens_total"] <= runtime["mtp_draft_tokens_total"]
    assert 0 < runtime["mtp_acceptance_rate"] <= 1

    assert gsm["sample_count"] == 1319
    assert 0.0 <= gsm["exact_match_flexible_extract"] <= 1.0
    rows = concurrency["rows"]
    assert [row["concurrency"] for row in rows] == [1, 2, 4, 8, 16, 32]
    assert all(row["completed"] > 0 and row["failed"] == 0 for row in rows)
    assert len(llama.get("benchmarks", [])) == 4
    assert telemetry.get("phases")
    assert len(energy.get("rows", [])) == 6

    required = [
        "Qwen3.6 35B-A3B", "vLLM 0.25.0", "FP8", "ModelOpt", "FlashInfer",
        "MTP K=2", "Concurrency scaling", "Mean active power W", "NVFP4-KV candidate",
    ]
    for item in required:
        assert item in page, f"HTML missing {item!r}"
    forbidden = ["NVFP4 Fast", "MARLIN backend selected", "vLLM 0.24"]
    for item in forbidden:
        assert item.lower() not in page.lower(), f"HTML contains forbidden value {item!r}"
    assert page.count("<div") == page.count("</div>")
    assert page.count("<section") == page.count("</section>")
    assert "opacity:0" not in page

    print("PASS NVIDIA Qwen vLLM v0.25 MTP release consistency gate")
    print(f"GSM8K={gsm['exact_match_flexible_extract']:.6f} samples={gsm['sample_count']}")
    print(f"MTP_acceptance={runtime['mtp_acceptance_rate']:.6f} concurrency_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
