#!/usr/bin/env python3
"""Patch vLLM ModelOpt mixed linear routing for this calibrated NVIDIA checkpoint.

The checkpoint labels shared-expert/LM-head linear layers W4A16_NVFP4 in
hf_quant_config.json, but also ships finite calibrated input_scale tensors and
declares 4-bit input activations in config.json. Upstream vLLM discards those
scales and hard-pins Marlin. This dedicated image instead routes only LinearBase
W4A16 layers through ModelOptNvFp4LinearMethod, which consumes the published
input scales and selects the native FlashInfer/CUTLASS W4A4 kernel. RoutedExperts
remain on the independent native MoE backend.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = "vllm.model_executor.layers.quantization.modelopt"
MARKER = "R0B0TLAB_NATIVE_W4A4_FROM_W4A16"
OLD = '''            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4W4A16LinearMethod(self.w4a16_nvfp4_config)
'''
NEW = f'''            if quant_algo == "W4A16_NVFP4":
                logger.warning_once(
                    "{MARKER}: routing calibrated W4A16-labelled linear "
                    "layers through native NVFP4 W4A4; checkpoint input_scale "
                    "tensors are required and consumed."
                )
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
'''

spec = importlib.util.find_spec(MODULE)
if spec is None or spec.origin is None:
    raise SystemExit(f"cannot locate {MODULE}")
path = Path(spec.origin)
text = path.read_text()
if MARKER in text:
    print(f"already patched: {path}")
    raise SystemExit(0)
if text.count(OLD) != 1:
    raise SystemExit(f"expected exactly one W4A16 linear routing block in {path}")
path.write_text(text.replace(OLD, NEW))
check = path.read_text()
if MARKER not in check or OLD in check:
    raise SystemExit("post-patch verification failed")
print(f"patched: {path}")
