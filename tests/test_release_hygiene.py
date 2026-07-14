from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicTreeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = load_module("public_tree_guard", ROOT / "scripts/public_tree_guard.py")

    def scan_fixture(self, files: dict[str, str]):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, content in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            return self.guard.scan_paths(root, sorted(p for p in root.rglob("*") if p.is_file()))

    def test_rejects_unused_speculative_decoder_name(self):
        forbidden = "d" + "flash"
        failures = self.scan_fixture({"README.md": f"Comparison against {forbidden}.\n"})
        self.assertTrue(any(item.rule == "unused_speculative_decoder" for item in failures))

    def test_rejects_private_and_raw_artifacts(self):
        private_ip = ".".join(("192", "168", "2", "2"))
        home_path = "/".join(("", "home", "operator", "model"))
        token = "gh" + "p_" + "abcdefghijklmnopqrstuvwxyz123456"
        failures = self.scan_fixture(
            {
                "README.md": f"host {private_ip} path {home_path} token {token}\n",
                "benchmarks/raw/sample.jsonl": "{}\n",
                "benchmarks/telemetry.jsonl": "{}\n",
                "weights/model-00001-of-00002.safetensors": "not-a-weight",
            }
        )
        rules = {item.rule for item in failures}
        self.assertTrue({"private_ipv4", "absolute_home", "credential", "raw_artifact", "model_or_cache"} <= rules)

    def test_allows_public_urls_and_curated_summaries(self):
        failures = self.scan_fixture(
            {
                "README.md": "https://github.com/r0b0tlab/project\nhttps://huggingface.co/nvidia/model\n",
                "benchmarks/telemetry-summary.json": json.dumps({"power_w_mean": 42.0}),
                "benchmarks/runtime-manifest.json": json.dumps({"image_id": "sha256:abc"}),
            }
        )
        self.assertEqual([], failures)


class PreflightTests(unittest.TestCase):
    def test_preflight_requires_explicit_output_and_writes_expected_identity(self):
        script = ROOT / "scripts/preflight_v025_mtp.py"
        missing = subprocess.run(["python3", str(script)], text=True, capture_output=True)
        self.assertNotEqual(0, missing.returncode)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preflight.json"
            run = subprocess.run(
                ["python3", str(script), "--output", str(output), "--root", str(ROOT)],
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(output.read_text())
            self.assertEqual("nvidia/Qwen3.6-35B-A3B-NVFP4", payload["model_repository"])
            self.assertEqual("491c2f1ea524c639598bf8fa787a93fed5a6fbce", payload["model_revision"])
            self.assertEqual(
                "sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5",
                payload["base_image_digest"],
            )
            self.assertTrue(payload["hostname"])
            self.assertTrue(payload["generated_at_utc"].endswith("+00:00"))
            self.assertIn("wrote private preflight", run.stdout)


if __name__ == "__main__":
    unittest.main()
