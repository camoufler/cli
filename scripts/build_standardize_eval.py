#!/usr/bin/env python3
"""Build tests/eval/standardize.json from scripts/standardize_raw.tsv."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "scripts" / "standardize_raw.tsv"
OUT_PATH = ROOT / "tests" / "eval" / "standardize.json"

INSTRUCTION_PREFIX = re.compile(
    r"^(?:"
    r"Remove all grammatical errors from this text: "
    r"|Improve the grammaticality(?: of this (?:text|sentence))?: "
    r"|Fix grammaticality(?: in this sentence| of the sentence| of this sentence)?: "
    r"|Fix grammar(?: in this sentence| in the sentence|:)? "
    r"|Update to remove grammar errors: "
    r"|Make the sentence (?:grammatical|fluent): "
    r"|Remove grammatical mistakes: "
    r"|Remove grammar mistakes: "
    r"|Grammar improvements: "
    r"|Improve the grammar of this text: "
    r"|Fix the grammatical mistakes: "
    r"|Fix grammatical mistakes in this sentence: "
    r"|Fix (?:all )?grammatical errors(?: in this sentence)?: "
    r"|Fix grammatical errors: "
    r"|Fix disfluencies in the sentence: "
    r"|Fix errors in this text: "
    r")",
    re.IGNORECASE,
)


def strip_prefix(raw: str) -> tuple[str, bool]:
    """Return (text, prefix_was_stripped)."""
    match = INSTRUCTION_PREFIX.match(raw)
    if match:
        return raw[match.end() :].strip(), True
    return raw.strip(), False


def main() -> int:
    if not RAW_PATH.is_file():
        print(f"error: missing {RAW_PATH}", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    warnings: list[str] = []

    for line_no, line in enumerate(RAW_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            warnings.append(f"line {line_no}: expected tab-separated pair, got {len(parts)} fields")
            continue
        source, reference = parts[0].strip(), parts[1].strip()
        if not source or not reference:
            warnings.append(f"line {line_no}: empty source or reference")
            continue
        text, stripped = strip_prefix(source)
        if not stripped:
            warnings.append(f"line {line_no}: no instruction prefix stripped — using raw source")
        if not text:
            warnings.append(f"line {line_no}: empty input after prefix strip")
            continue
        rows.append({"input": text, "reference": reference})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} entries to {OUT_PATH}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
