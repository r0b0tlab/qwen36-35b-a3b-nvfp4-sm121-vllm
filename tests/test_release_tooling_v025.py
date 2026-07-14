from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = "d" + "flash"


def source(rel: str) -> str:
    return (ROOT / rel).read_text()


class ReleaseToolingTests(unittest.TestCase):
    def test_capture_records_v025_image_and_native_markers(self):
        capture = source("scripts/capture_release.py")
        for required in (
            "FlashInferFP8ScaledMMLinearKernel",
            "FlashInferCutlassNvFp4LinearKernel",
            "R0B0TLAB_NATIVE_W4A4_FROM_W4A16",
            "FLASHINFER_B12X",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
            "base_image_digest",
            "image_labels",
        ):
            self.assertIn(required, capture)
        self.assertNotIn(FORBIDDEN, capture.lower())

    def test_release_verifier_requires_full_v025_contract(self):
        verify = source("scripts/verify_release.py")
        for required in (
            'runtime["runtime"]["vllm"] == "0.25.0"',
            'runtime["runtime"]["torch"] == "2.11.0+cu130"',
            'runtime["runtime"]["cuda"] == "13.0"',
            'runtime["runtime"]["flashinfer"] == "0.6.13"',
            "durability.summary.json",
            "runtime-audit.json",
            "w4a16-input-scale-audit.json",
            "mtp_acceptance_rate",
        ):
            self.assertIn(required, verify)

    def test_report_identifies_v025_mtp(self):
        report = source("scripts/generate_report.py")
        self.assertIn("vLLM 0.25.0", report)
        self.assertIn("MTP K=2", report)
        self.assertNotIn("vLLM 0.24", report)
        self.assertNotIn(FORBIDDEN, report.lower())

    def test_raw_durability_is_ignored_and_manifest_excluded(self):
        self.assertIn("durability.jsonl", source(".gitignore"))
        manifest = source("scripts/write_manifest.py")
        self.assertIn("durability.jsonl", manifest)
        self.assertIn("telemetry.jsonl", manifest)


if __name__ == "__main__":
    unittest.main()
