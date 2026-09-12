"""download and standardize command handlers."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from camoufler import __version__
from camoufler.config import AppConfig, cpu_options
from camoufler.models import ensure_small
from camoufler.ollama_api import pull_model
from camoufler.pipeline import run_standardize

if TYPE_CHECKING:
    import argparse


def read_stdin() -> str:
    """Read all stdin; error if empty."""
    text = sys.stdin.read()
    if not text.strip():
        raise ValueError("No input text on stdin.")
    return text.strip()


def cmd_download(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Download the specified small Ollama model."""
    model = ensure_small(args.model)
    logger.info("Downloading %s", model)
    pull_model(model)
    logger.info("Download complete: %s", model)


def cmd_standardize(
    args: argparse.Namespace,
    config: AppConfig,
    logger: logging.Logger,
) -> str:
    """Detect prompt type, then rewrite or expand stdin text."""
    model = ensure_small(args.model)
    user_text = read_stdin()
    opts = cpu_options(config.options)
    logger.info("Standardizing with %s", model)
    return run_standardize(model, config, user_text, opts)


def cmd_standardize_interactive(
    args: argparse.Namespace,
    config: AppConfig,
    logger: logging.Logger,
) -> None:
    """REPL: Enter submits one line; Ctrl+C or EOF quits."""
    model = ensure_small(args.model)
    opts = cpu_options(config.options)
    print(
        f"camoufler {__version__}  model {model}\n"
        "You type under #user. Camoufler answers under #camoufler.\n"
        "Enter to send. Ctrl+C or Ctrl+D to quit.\n",
        file=sys.stderr,
    )
    while True:
        print("#user", file=sys.stderr)
        sys.stderr.flush()
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.", file=sys.stderr)
            return
        user_text = line.strip()
        if not user_text:
            continue
        logger.info("Standardizing with %s", model)
        try:
            result = run_standardize(model, config, user_text, opts)
        except Exception as exc:  # noqa: BLE001 — keep the REPL alive
            print(f"\n#camoufler\nerror: {exc}\n", file=sys.stderr)
            continue
        print("\n#camoufler", file=sys.stderr)
        sys.stderr.flush()
        print(result)
        print()
        sys.stdout.flush()
