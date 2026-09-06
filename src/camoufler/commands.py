"""download and standardize command handlers."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

from camoufler.config import AppConfig, cpu_options
from camoufler.models import ensure_small
from camoufler.ollama_api import chat_standardize, pull_model

if TYPE_CHECKING:
    import argparse


def _ensure_readline() -> None:
    """Load readline on supported platforms so line-buffer queries work."""
    try:
        import readline  # noqa: F401
    except ImportError:
        pass


def _partial_line() -> str:
    """Return the in-progress line when Ctrl+C interrupts readline."""
    try:
        import readline

        return readline.get_line_buffer()
    except (ImportError, AttributeError, OSError):
        return ""


def read_stdin() -> str:
    """Read all stdin; error if empty."""
    text = sys.stdin.read()
    if not text.strip():
        raise ValueError("No input text on stdin.")
    return text.strip()


def iter_interactive_inputs() -> Iterator[str]:
    """Yield text chunks submitted by Ctrl+C on a TTY; EOF exits."""
    _ensure_readline()
    buffer: list[str] = []
    while True:
        try:
            line = sys.stdin.readline()
            if line == "":
                return
            buffer.append(line.rstrip("\n\r"))
        except KeyboardInterrupt:
            partial = _partial_line().strip()
            if partial:
                buffer.append(partial)
            text = "\n".join(buffer).strip()
            buffer = []
            sys.stdout.write("\n")
            sys.stdout.flush()
            if text:
                yield text


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
    """Standardize stdin text using the configured model and prompt."""
    model = ensure_small(args.model)
    user_text = read_stdin()
    opts = cpu_options(config.options)
    logger.info("Standardizing with %s", model)
    return chat_standardize(model, config.system_prompt, user_text, opts)


def cmd_standardize_interactive(
    args: argparse.Namespace,
    config: AppConfig,
    logger: logging.Logger,
) -> None:
    """Standardize interactive TTY input; Ctrl+C submits, EOF exits."""
    model = ensure_small(args.model)
    opts = cpu_options(config.options)
    for user_text in iter_interactive_inputs():
        logger.info("Standardizing with %s", model)
        result = chat_standardize(model, config.system_prompt, user_text, opts)
        print(result)
        sys.stdout.flush()
