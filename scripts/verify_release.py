#!/usr/bin/env python3
"""Consistency gate for the streamlined vLLM v0.25 MTP release."""
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
    page = Path(args.html).read_text()
    reuse = load(run / "evidence-reuse.json")
    equivalence = load(run / "equivalence/summary.json")
    gsm = load(run / "gsm8k/results.json")
    concurrency = load(run / "concurrency/summary.json")
    runtime = load(run / "runtime-manifest.json")
    audit = load(run / "evidence/runtime-audit.json")
    scales = load(run / "evidence/w4a16-input-scale-audit-clean.json")
    semantic = load(run / "equivalence/semantic.json")
    long_generation = load(run / "equivalence/long-generation.json")
    maxctx = load(run / "max-context.json")

    assert reuse["passed"] and all(reuse["checks"].values())
    assert equivalence["passed"] and all(equivalence["checks"].values())
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
    assert maxctx["status"] == "pass"
    assert maxctx["architectural_context_tokens"] == 262144
    assert maxctx["base_ar"]["validated_context_tokens"] == 262144
    assert maxctx["mtp_k2"]["validated_context_tokens"] == 262144
    assert maxctx["base_ar"]["measured_kv_capacity_tokens"] >= 262144
    assert maxctx["mtp_k2"]["measured_kv_capacity_tokens"] >= 262144
    assert maxctx["base_ar"]["near_window"]["positions_passed"] == ["begin", "quarter", "middle", "three_quarter", "end"]
    assert maxctx["mtp_k2"]["near_window"]["positions_passed"] == ["begin", "quarter", "middle", "three_quarter", "end"]
    assert maxctx["base_ar"]["near_window"]["forced_generation_tokens"] == 512
    assert maxctx["mtp_k2"]["near_window"]["forced_generation_tokens"] == 512
    assert 0 < maxctx["mtp_k2"]["acceptance_rate"] <= 1
    assert maxctx["base_ar"]["added_swap_mib"] == 0 and maxctx["mtp_k2"]["added_swap_mib"] == 0

    profile = runtime["profile"]
    assert profile["KV_CACHE_DTYPE"] == "fp8"
    assert profile["QUANTIZATION"] == "modelopt_mixed"
    assert profile["MOE_BACKEND"] == "flashinfer_b12x"
    assert profile["LINEAR_BACKEND"] == "flashinfer_cutlass"
    assert profile["ATTENTION_BACKEND"] == "flashinfer"
    assert json.loads(profile["SPECULATIVE_CONFIG"]) == {
        "method": "mtp", "num_speculative_tokens": 2, "moe_backend": "triton"
    }
    required_markers = {
        "modelopt_mixed", "fp8_dense", "nvfp4_dense", "w4a16_native_reroute",
        "target_moe_native", "flashinfer_attention", "effective_piecewise", "mtp_configured",
    }
    assert all(runtime["native_marker_gate"].get(key) for key in required_markers)
    assert 0 < runtime["mtp_acceptance_rate"] <= 1

    assert gsm["sample_count"] == 1319
    assert gsm["model_args"]["max_retries"] == 0
    assert 0.0 <= gsm["exact_match_flexible_extract"] <= 1.0
    rows = concurrency["rows"]
    assert [row["concurrency"] for row in rows] == [1, 8, 32]
    assert all(row["repetitions"] == 3 and row["failed"] == 0 for row in rows)

    for item in ("Qwen3.6 35B-A3B", "vLLM 0.25.0", "FP8", "ModelOpt", "FlashInfer", "MTP K=2", "262,144 tokens validated", "261,883 tokens"):
        assert item in page, f"HTML missing {item!r}"
    for item in ("NVFP4 Fast", "MARLIN backend selected", "vLLM 0.24"):
        assert item.lower() not in page.lower(), f"HTML contains forbidden value {item!r}"
    assert page.count("<div") == page.count("</div>")
    assert page.count("<section") == page.count("</section>")
    print("PASS streamlined vLLM v0.25 MTP release consistency gate")
    print(f"GSM8K={gsm['exact_match_flexible_extract']:.6f} samples={gsm['sample_count']}")
    print(f"MTP_acceptance={runtime['mtp_acceptance_rate']:.6f} points={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
