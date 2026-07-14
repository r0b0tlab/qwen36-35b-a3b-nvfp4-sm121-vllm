from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_REF = (
    "ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121:"
    "v0.25.0-cu130-sm121-arm64-702f4814-r2@"
    "sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5"
)
FORBIDDEN = "d" + "flash"


def text(rel: str) -> str:
    return (ROOT / rel).read_text()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DockerfileTests(unittest.TestCase):
    def test_digest_pinned_mtp_only_runtime(self):
        source = text("Dockerfile.v025")
        self.assertIn(f"FROM {BASE_REF}", source)
        self.assertIn("702f4814fe54fabff350d43cb753ae3e47c0c276", source)
        self.assertIn("patch_modelopt_w4a16_native_w4a4.py", source)
        self.assertIn("audit_runtime_v025.py", source)
        self.assertIn("start_vllm_v025.sh", source)
        self.assertIn('SPECULATIVE_CONFIG=\'{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}\'', source)
        self.assertNotIn(FORBIDDEN, source.lower())

    def test_no_extra_runtime_patch_payloads(self):
        copy_lines = [line.strip() for line in text("Dockerfile.v025").splitlines() if line.strip().startswith("COPY patches/")]
        self.assertEqual(
            ["COPY patches/patch_modelopt_w4a16_native_w4a4.py /tmp/patch_modelopt_w4a16_native_w4a4.py"],
            copy_lines,
        )


class EntrypointTests(unittest.TestCase):
    def test_exact_runtime_defaults(self):
        source = text("scripts/start_vllm_v025.sh")
        for expected in (
            'KV_CACHE_DTYPE:-fp8',
            'ATTENTION_BACKEND:-flashinfer',
            'MOE_BACKEND:-flashinfer_b12x',
            'LINEAR_BACKEND:-flashinfer_cutlass',
            'GPU_MEMORY_UTILIZATION:-0.88',
            'MAX_MODEL_LEN:-65536',
            'MAX_NUM_SEQS:-32',
            'MAX_NUM_BATCHED_TOKENS:-32768',
            'QUANTIZATION:-modelopt_mixed',
        ):
            self.assertIn(expected, source)
        self.assertIn("--speculative-config", source)
        self.assertNotIn(FORBIDDEN, source.lower())


class AuditTests(unittest.TestCase):
    def test_audit_pins_full_runtime(self):
        source = text("scripts/audit_runtime_v025.py")
        for expected in (
            '"vllm": "0.25.0"',
            '"flashinfer-python": "0.6.13"',
            '"nvidia-cutlass-dsl": "4.5.2"',
            '"nvidia-nccl-cu13": "2.28.9"',
            '"2.11.0+cu130"',
            '"13.0"',
            'vllm.model_executor.models.qwen3_5_mtp',
            'R0B0TLAB_NATIVE_W4A4_FROM_W4A16',
            'cutlass_scaled_mm_supports_fp4',
        ):
            self.assertIn(expected, source)
        self.assertNotIn(FORBIDDEN, source.lower())


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            "patch_modelopt_w4a16_native_w4a4",
            ROOT / "patches/patch_modelopt_w4a16_native_w4a4.py",
        )

    def test_patch_is_narrow_and_idempotent(self):
        fixture = "prefix\n" + self.module.OLD + "suffix\n"
        patched, status = self.module.patch_text(fixture)
        self.assertEqual("patched", status)
        self.assertIn(self.module.MARKER, patched)
        self.assertNotIn(self.module.OLD, patched)
        again, status = self.module.patch_text(patched)
        self.assertEqual("already-patched", status)
        self.assertEqual(patched, again)

    def test_patch_rejects_unknown_dispatcher(self):
        with self.assertRaises(ValueError):
            self.module.patch_text("unrelated source")


class ComposeAndBuildTests(unittest.TestCase):
    def test_compose_and_build_use_v025_mtp_image(self):
        compose = text("docker-compose.yml")
        build = text("scripts/build_v025_image.sh")
        self.assertIn("Dockerfile.v025", compose)
        self.assertIn("Dockerfile.v025", build)
        self.assertIn("v025-mtp", compose)
        self.assertIn("v025-mtp", build)
        self.assertNotIn(FORBIDDEN, (compose + build).lower())


if __name__ == "__main__":
    unittest.main()
