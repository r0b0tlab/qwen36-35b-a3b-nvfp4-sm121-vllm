#!/usr/bin/env python3
"""Atomic, no-retry sustained decode gate for the accepted MTP profile."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen3.6-35B-A3B-NVFP4"
PROMPTS = (
    "Explain why FP8 KV cache and NVFP4 weights solve different memory problems.",
    "Compute 1237 times 43 and end with the integer result.",
    "Write a concise Python function that validates a SHA-256 hexadecimal string.",
    "Describe three failure modes in speculative decoding acceptance metrics.",
)


def append_atomic(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=250)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite durability evidence: {output}")
    passed = 0
    for index in range(args.requests):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        row = {"index": index, "request": payload}
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                body = json.loads(response.read())
                choice = (body.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                row.update({
                    "http_status": response.status,
                    "elapsed_seconds": time.perf_counter() - started,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage"),
                    "content": content,
                    "passed": response.status == 200 and bool(content.strip()),
                })
        except Exception as exc:
            row.update({
                "http_status": 0,
                "elapsed_seconds": time.perf_counter() - started,
                "error": repr(exc),
                "passed": False,
            })
        append_atomic(output, row)
        if not row["passed"]:
            print(json.dumps({"passed": False, "failed_index": index, "output": str(output)}))
            return 1
        passed += 1
    summary = {"passed": True, "completed": passed, "requested": args.requests}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
