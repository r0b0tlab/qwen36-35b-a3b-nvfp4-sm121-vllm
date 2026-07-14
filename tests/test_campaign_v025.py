from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(rel: str) -> str:
    return (ROOT / rel).read_text()


class CampaignContractTests(unittest.TestCase):
    def test_full_campaign_reruns_all_correctness_gates(self):
        script = source("scripts/run_full_campaign.sh")
        self.assertNotIn("fp8-native-w4a4-v1", script)
        self.assertNotIn("cp \"$BASELINE", script)
        for required in (
            "audit_runtime_v025.py",
            "audit_w4a16_input_scales.py",
            "run_semantic_gate.py",
            "run_long_generation.py",
            "run_gsm8k.sh",
            "run_concurrency.sh",
            "run_llama_benchy.sh",
            "run_durability.py",
            "finalize_release.sh",
        ):
            self.assertIn(required, script)
        self.assertIn("--requests 250", script)
        self.assertLess(script.index("docker stop --time 30"), script.index("campaign-complete"))

    def test_shell_helpers_are_invoked_through_bash(self):
        for rel in ("scripts/run_full_campaign.sh", "scripts/finalize_release.sh"):
            script = source(rel)
            direct = re.findall(r'^\s*"\$ROOT/scripts/[^\"]+\.sh"', script, re.MULTILINE)
            self.assertEqual([], direct, f"direct non-portable invocation in {rel}: {direct}")

    def test_gsm8k_is_fail_closed_and_exact(self):
        script = source("scripts/run_gsm8k.sh")
        self.assertIn("num_concurrent=1", script)
        self.assertIn("max_retries=0", script)
        self.assertIn("--num_fewshot 0", script)
        self.assertIn("--apply_chat_template", script)
        self.assertIn('"enable_thinking":false', script)
        self.assertIn('"max_gen_toks":2048', script)
        self.assertIn("BENCH_PY", script)

    def test_concurrency_is_seeded(self):
        self.assertIn("--seed 0", source("scripts/run_concurrency.sh"))

    def test_launch_defaults_to_v025_mtp(self):
        launch = source("scripts/launch.sh")
        self.assertIn("v025-mtp", launch)
        self.assertIn("ATTENTION_BACKEND", launch)
        self.assertIn("SPECULATIVE_CONFIG", launch)
        self.assertNotIn("docker rm -f", launch)


if __name__ == "__main__":
    unittest.main()
