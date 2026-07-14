#!/usr/bin/env python3
"""Fail closed when candidate public files contain private or out-of-scope data."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class Finding(NamedTuple):
    path: str
    rule: str
    detail: str


UNUSED_SPECULATIVE_DECODER = "d" + "flash"
TEXT_PATTERNS = {
    "unused_speculative_decoder": re.compile(UNUSED_SPECULATIVE_DECODER, re.IGNORECASE),
    "private_ipv4": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
    "absolute_home": re.compile(r"/home/[A-Za-z0-9_.-]+(?:/[^\s`'\"]*)?"),
    "credential": re.compile(
        r"\b(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})\b"
    ),
}
RAW_NAMES = {
    "telemetry.jsonl",
    "progress.jsonl",
    "samples.jsonl",
    "run.log",
    "server.log",
    "campaign.log",
    "telemetry.pid",
}
MODEL_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}
CACHE_PARTS = {".cache", "__pycache__", "flashinfer_cache", "torch_compile_cache"}


def path_rule(rel: Path) -> str | None:
    parts_lower = {part.lower() for part in rel.parts}
    name = rel.name.lower()
    if "raw" in parts_lower or name in RAW_NAMES or name.endswith(".log") or name.endswith(".pid"):
        return "raw_artifact"
    if rel.suffix.lower() == ".jsonl" and name != "telemetry-summary.json":
        return "raw_artifact"
    if rel.suffix.lower() in MODEL_SUFFIXES or parts_lower & CACHE_PARTS or rel.suffix.lower() in {".pyc", ".pyo"}:
        return "model_or_cache"
    return None


def scan_paths(root: Path, paths: list[Path]) -> list[Finding]:
    failures: list[Finding] = []
    for path in paths:
        try:
            rel = path.resolve().relative_to(root.resolve())
        except ValueError:
            failures.append(Finding(str(path), "outside_root", "candidate is outside repository"))
            continue
        rule = path_rule(rel)
        if rule:
            failures.append(Finding(str(rel), rule, rel.name))
            continue
        if path.stat().st_size > 100_000_000:
            failures.append(Finding(str(rel), "oversized", str(path.stat().st_size)))
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            failures.append(Finding(str(rel), "unreadable", str(exc)))
            continue
        for name, pattern in TEXT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                failures.append(Finding(str(rel), name, match.group(0)))
    return failures


def candidate_paths(root: Path) -> list[Path]:
    if not (root / ".git").exists() and not subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-dir"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0:
        return sorted(path for path in root.rglob("*") if path.is_file())
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"],
    ).decode().split("\0")
    staged = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
    ).decode().split("\0")
    rels = sorted({item for item in tracked + staged if item})
    return [root / rel for rel in rels if (root / rel).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    failures = scan_paths(root, candidate_paths(root))
    if failures:
        for finding in failures:
            print("FAIL", finding.path, finding.rule, finding.detail)
        return 1
    print(f"PASS public tree guard files={len(candidate_paths(root))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
