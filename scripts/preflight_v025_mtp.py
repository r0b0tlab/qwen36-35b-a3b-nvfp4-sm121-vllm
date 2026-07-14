#!/usr/bin/env python3
"""Capture private, run-scoped preflight evidence for the vLLM v0.25 MTP release."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import subprocess
from pathlib import Path

MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
BASE_IMAGE = "ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121:v0.25.0-cu130-sm121-arm64-702f4814-r2"
BASE_IMAGE_DIGEST = "sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Private output JSON path")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise SystemExit("preflight output must be outside the repository")
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "repository_root": str(root),
        "git_commit": git(root, "rev-parse", "HEAD"),
        "git_branch": git(root, "branch", "--show-current"),
        "git_remote": git(root, "remote", "get-url", "origin"),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "base_image": BASE_IMAGE,
        "base_image_digest": BASE_IMAGE_DIGEST,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    output.chmod(0o600)
    print(f"wrote private preflight: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
