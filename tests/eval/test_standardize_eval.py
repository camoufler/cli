"""Integration eval for standardize against eval JSON datasets."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from sacrebleu.metrics import CHRF

from camoufler.commands import cmd_standardize
from camoufler.config import load_config

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.example.json"
ASK_FRAMING_CONFIG_PATH = ROOT / "config" / "config.example2.json"

CHRF_THRESHOLD = 0.45
PASS_RATE_THRESHOLD = 0.75
DEFAULT_MODEL = "qwen2.5:1.5b"
# Cap rows per suite (override with CAMOUFLER_EVAL_CAP). Ask framing defaults lower.
DEFAULT_EVAL_CAP = 100
ASK_FRAMING_EVAL_CAP = 10

_chrf = CHRF()


def _ollama_available() -> bool:
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


def _chrf_score(prediction: str, reference: str) -> float:
    return _chrf.sentence_score(prediction, [reference]).score / 100.0


def _args() -> argparse.Namespace:
    return argparse.Namespace(model=DEFAULT_MODEL, verbose=0)


def _eval_cap(default: int) -> int:
    raw = os.environ.get("CAMOUFLER_EVAL_CAP")
    if raw is None or not raw.strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _load_eval(path: Path, *, cap: int | None = None) -> list[dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    limit = _eval_cap(cap if cap is not None else DEFAULT_EVAL_CAP)
    return rows[:limit]


def _run_eval(name: str, rows: list[dict[str, str]], app_config) -> None:
    logger = logging.getLogger(f"{name}-eval")
    passed = 0
    failures: list[str] = []
    scores: list[float] = []

    print(f"\n=== {name} eval ({len(rows)} rows) ===")
    for index, row in enumerate(rows, 1):
        ask = row["input"]
        expected = row["reference"]
        try:
            with patch("camoufler.commands.read_stdin", return_value=ask):
                prediction = cmd_standardize(_args(), app_config, logger)
        except Exception as exc:  # noqa: BLE001 — eval soft-fail per row
            print(f"\n#{index} FAIL (error)")
            print(f"  ask:      {ask}")
            print(f"  response: <error: {exc}>")
            print(f"  expected: {expected}")
            print(f"  score:    n/a")
            failures.append(
                f"#{index} error={exc}\n"
                f"  input: {ask[:120]}{'...' if len(ask) > 120 else ''}\n"
                f"  expected: {expected[:120]}{'...' if len(expected) > 120 else ''}"
            )
            continue

        score = _chrf_score(prediction, expected)
        scores.append(score)
        ok = score >= CHRF_THRESHOLD
        if ok:
            passed += 1
        else:
            failures.append(
                f"#{index} chrF={score:.3f}\n"
                f"  input: {ask[:120]}{'...' if len(ask) > 120 else ''}\n"
                f"  expected: {expected[:120]}{'...' if len(expected) > 120 else ''}\n"
                f"  got: {prediction[:120]}{'...' if len(prediction) > 120 else ''}"
            )

        print(f"\n#{index} {'PASS' if ok else 'FAIL'} (chrF={score:.3f})")
        print(f"  ask:      {ask}")
        print(f"  response: {prediction}")
        print(f"  expected: {expected}")
        print(f"  score:    {score:.3f} (threshold {CHRF_THRESHOLD})")

    pass_rate = passed / len(rows)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    summary = (
        f"{name} eval: {passed}/{len(rows)} passed ({pass_rate:.0%}), "
        f"avg chrF={avg_score:.3f}"
    )
    print(f"\n--- {summary} ---")
    if pass_rate < PASS_RATE_THRESHOLD:
        detail = summary + "\n\nFailures:\n" + "\n\n".join(failures[:10])
        if len(failures) > 10:
            detail += f"\n\n... and {len(failures) - 10} more"
        pytest.fail(detail)


@pytest.fixture(scope="module")
def app_config():
    return load_config(str(CONFIG_PATH), required=True)


@pytest.fixture(scope="module")
def ask_framing_config():
    return load_config(str(ASK_FRAMING_CONFIG_PATH), required=True)


def test_standardize_eval(app_config):
    if not _ollama_available():
        pytest.skip("Ollama not available")
    rows = _load_eval(ROOT / "tests" / "eval" / "standardize.json")
    _run_eval("standardize", rows, app_config)


def test_slang_standardize_eval(app_config):
    if not _ollama_available():
        pytest.skip("Ollama not available")
    rows = _load_eval(ROOT / "tests" / "eval" / "slang_standardize.json")
    _run_eval("slang", rows, app_config)


def test_ask_framing_eval(ask_framing_config):
    if not _ollama_available():
        pytest.skip("Ollama not available")
    rows = _load_eval(
        ROOT / "tests" / "eval" / "ask_framing.json",
        cap=ASK_FRAMING_EVAL_CAP,
    )
    _run_eval("ask_framing", rows, ask_framing_config)


def test_prompt_expand_eval(app_config):
    if not _ollama_available():
        pytest.skip("Ollama not available")
    rows = _load_eval(ROOT / "tests" / "eval" / "prompt_expand.json")
    _run_eval("prompt_expand", rows, app_config)
