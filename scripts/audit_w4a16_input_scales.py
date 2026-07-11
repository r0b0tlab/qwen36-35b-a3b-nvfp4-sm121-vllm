#!/usr/bin/env python3
"""Audit calibrated input scales required by the native W4A4 linear reroute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.model_path)
    hf = json.loads((root / "hf_quant_config.json").read_text())
    qlayers = hf["quantization"]["quantized_layers"]
    w4a16 = sorted(k for k, v in qlayers.items() if v.get("quant_algo") == "W4A16_NVFP4")
    routed = [k for k in w4a16 if k.endswith(".mlp.experts")]
    linear = [k for k in w4a16 if k not in routed]
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    required_suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    missing = [f"{prefix}.{suffix}" for prefix in linear for suffix in required_suffixes if f"{prefix}.{suffix}" not in index]
    by_file: dict[str, list[str]] = {}
    for prefix in linear:
        key = prefix + ".input_scale"
        if key in index:
            by_file.setdefault(index[key], []).append(key)

    values: list[float] = []
    bad: list[dict] = []
    shapes: dict[str, list[int]] = {}
    for filename, keys in by_file.items():
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                tensor = handle.get_tensor(key).float().reshape(-1)
                shapes[key] = list(tensor.shape)
                if not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all()):
                    bad.append({"key": key, "values": tensor.tolist()})
                values.extend(tensor.tolist())

    result = {
        "model_path": str(root),
        "w4a16_targets": len(w4a16),
        "routed_expert_aggregate_targets": len(routed),
        "linear_targets": len(linear),
        "linear_input_scales_loaded": len(shapes),
        "required_tensor_keys_missing": missing,
        "nonfinite_or_nonpositive_scales": bad,
        "scale_count": len(values),
        "scale_min": min(values) if values else None,
        "scale_max": max(values) if values else None,
        "expected_contract": {
            "w4a16_targets": 161,
            "routed_expert_aggregate_targets": 40,
            "linear_targets": 121,
            "linear_input_scales_loaded": 121,
        },
    }
    result["passed"] = (
        result["w4a16_targets"] == 161
        and result["routed_expert_aggregate_targets"] == 40
        and result["linear_targets"] == 121
        and result["linear_input_scales_loaded"] == 121
        and not missing
        and not bad
        and len(values) == 121
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
