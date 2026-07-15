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

    def test_release_verifier_requires_reuse_and_clean_image_contract(self):
        verify = source("scripts/verify_release.py")
        for required in (
            'runtime["runtime"]["vllm"] == "0.25.0"',
            'runtime["runtime"]["torch"] == "2.11.0+cu130"',
            'runtime["runtime"]["cuda"] == "13.0"',
            'runtime["runtime"]["flashinfer"] == "0.6.13"',
            "evidence-reuse.json",
            "equivalence/summary.json",
            "runtime-audit.json",
            "w4a16-input-scale-audit-clean.json",
            "mtp_acceptance_rate",
            "max-context.json",
            "architectural_context_tokens",
            "forced_generation_tokens",
        ):
            self.assertIn(required, verify)

    def test_report_identifies_v025_mtp(self):
        report = source("scripts/generate_report.py")
        self.assertIn("vLLM 0.25.0", report)
        self.assertIn("MTP K=2", report)
        self.assertNotIn("vLLM 0.24", report)
        self.assertNotIn(FORBIDDEN, report.lower())

    def test_documentation_generator_is_evidence_driven(self):
        generator = source("scripts/generate_release_docs.py")
        for required in (
            "gsm8k/results.json",
            "concurrency/summary.json",
            "spec-metrics.json",
            "runtime-manifest.json",
            "equivalence/summary.json",
            "MANIFEST.sha256",
            "evidence-reuse.json",
            "max-context.json",
        ):
            self.assertIn(required, generator)
        self.assertNotIn(FORBIDDEN, generator.lower())

    def test_max_context_reproduction_contract(self):
        launch = source("scripts/launch_max_context.sh")
        gate = source("scripts/run_max_context_gate.py")
        evidence = source("benchmarks/runs/qwen36-v025-mtp-20260714/max-context.json")
        for required in ("KV_CACHE_MEMORY_BYTES:-6G", "MAX_NUM_SEQS:-1", "--kv-cache-memory-bytes"):
            self.assertIn(required, launch)
        for required in ("begin", "quarter", "middle", "three_quarter", "end", "dual_code", "forced_512", "/tokenize"):
            self.assertIn(required, gate)
        for required in ('"architectural_context_tokens": 262144', '"validated_context_tokens": 262144', '"forced_generation_tokens": 512'):
            self.assertIn(required, evidence)

    def test_raw_durability_is_ignored_and_manifest_excluded(self):
        self.assertIn("durability.jsonl", source(".gitignore"))
        manifest = source("scripts/write_manifest.py")
        self.assertIn("durability.jsonl", manifest)
        self.assertIn("telemetry.jsonl", manifest)


if __name__ == "__main__":
    unittest.main()
