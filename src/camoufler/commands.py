"""download and standardize command handlers."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from camoufler.config import AppConfig, cpu_options
from camoufler.models import ensure_small
from camoufler.ollama_api import chat_standardize, pull_model

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
    """Standardize stdin text using the configured model and prompt."""
    model = ensure_small(args.model)
    user_text = read_stdin()
    opts = cpu_options(config.options)
    logger.info("Standardizing with %s", model)
    return chat_standardize(model, config.system_prompt, user_text, opts)
