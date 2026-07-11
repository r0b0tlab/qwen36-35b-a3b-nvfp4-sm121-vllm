#!/usr/bin/env python3
"""Fail-fast audit for NVIDIA Qwen ModelOpt mixed serving on GB10/SM121."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path


def check(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    return name, bool(ok), detail


def first_import(candidates: tuple[str, ...]) -> tuple[bool, str]:
    for name in candidates:
        try:
            importlib.import_module(name)
            return True, name
        except Exception:
            pass
    return False, ", ".join(candidates)


def main() -> int:
    import torch
    import vllm

    rows: list[tuple[str, bool, str]] = []
    version = getattr(vllm, "__version__", "unknown")
    rows.append(check("vllm_0.24", "0.24" in version, version))
    capability = torch.cuda.get_device_capability()
    rows.append(check("cuda_capability_sm121", capability == (12, 1), str(capability)))
    rows.append(check("cuda_runtime_13.x", str(torch.version.cuda).startswith("13."), str(torch.version.cuda)))

    for module in (
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm.model_executor.models.qwen3_5",
        "vllm.model_executor.models.qwen3_5_mtp",
        "vllm.v1.attention.backends.flashinfer",
    ):
        try:
            importlib.import_module(module)
            rows.append(check(f"import:{module}", True))
        except Exception as exc:
            rows.append(check(f"import:{module}", False, repr(exc)[:180]))

    modelopt_ok, modelopt_detail = first_import(
        (
            "vllm.model_executor.layers.quantization.modelopt",
            "vllm.model_executor.layers.quantization.modelopt_fp4",
            "vllm.model_executor.layers.quantization.modelopt_mixed",
        )
    )
    rows.append(check("modelopt_mixed_loader_module", modelopt_ok, modelopt_detail))
    if modelopt_ok:
        modelopt_spec = importlib.util.find_spec(modelopt_detail)
        modelopt_source = Path(modelopt_spec.origin).read_text(errors="replace") if modelopt_spec and modelopt_spec.origin else ""
        rows.append(check(
            "checkpoint_native_w4a4_reroute",
            "R0B0TLAB_NATIVE_W4A4_FROM_W4A16" in modelopt_source,
            modelopt_spec.origin if modelopt_spec and modelopt_spec.origin else "module source unavailable",
        ))

    try:
        support = bool(torch.ops._C.cutlass_scaled_mm_supports_fp4(121))
    except Exception as exc:
        support = False
        rows.append(check("sm121_cutlass_fp4", False, repr(exc)[:180]))
    else:
        rows.append(check("sm121_cutlass_fp4", support, str(support)))

    if os.getenv("REQUIRE_NVFP4_KV") == "1":
        quant_source = Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/nvfp4_cache_quant.py"
        )
        source = quant_source.read_text(errors="replace") if quant_source.exists() else ""
        rows.append(check("nvfp4_kv_quantizer_present", "nvfp4_quantize_and_cache" in source))
        rows.append(check("nvfp4_kv_real_block_scales", "amax" in source and "scale" in source))
        rows.append(check("nvfp4_kv_scale_factors_written", "sf_region" in source and "k_sf" in source))

    failed = [row for row in rows if not row[1]]
    print("AUDIT PASS" if not failed else "AUDIT FAIL")
    for name, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
