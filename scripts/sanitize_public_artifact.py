#!/usr/bin/env python3
"""Redact local identity, network, and credential strings in text artifacts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

RULES = (
    (re.compile(r"/home/[A-Za-z0-9_.-]+"), "${HOME}"),
    (
        re.compile(
            r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3}|"
            r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
        ),
        "[PRIVATE_IP]",
    ),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "[REDACTED]"),
    (re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"), "[REDACTED]"),
)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: sanitize_public_artifact.py FILE [FILE ...]")
    for raw in sys.argv[1:]:
        path = Path(raw)
        text = path.read_text(errors="replace")
        for pattern, replacement in RULES:
            text = pattern.sub(replacement, text)
        path.write_text(text)
        print(f"sanitized {path}")


if __name__ == "__main__":
    main()
