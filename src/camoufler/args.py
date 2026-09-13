"""CLI argument parsing."""

from __future__ import annotations

import argparse

from camoufler import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the camoufler argument parser."""
    parser = argparse.ArgumentParser(
        prog="camoufler",
        description="Download small local Ollama models and standardize English prompts.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"camoufler {__version__}",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Fully qualified Ollama model name (name:tag). "
        "Overrides the saved default when set.",
    )
    parser.add_argument(
        "-f",
        "--function",
        choices=["download", "list", "set", "standardize"],
        help="Operation. Omit on a terminal to open the menu.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Log level: 0=errors, 1=info, 2=debug.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="JSON config with system_prompt and Ollama options.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Function is optional (TTY menu)."""
    return build_parser().parse_args(argv)
