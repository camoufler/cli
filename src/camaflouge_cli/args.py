"""CLI argument parsing."""

from __future__ import annotations

import argparse

from camaflouge_cli import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the camaflouge-cli argument parser."""
    parser = argparse.ArgumentParser(
        prog="camaflouge-cli",
        description="Download small local Ollama models and standardize English text.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"camaflouge-cli {__version__}",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="qwen2.5:1.5b",
        help="Fully qualified Ollama model name (name:tag).",
    )
    parser.add_argument(
        "-f",
        "--function",
        choices=["download", "standardize"],
        help="Operation: download model or standardize stdin text.",
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
    """Parse CLI arguments and validate required fields."""
    args = build_parser().parse_args(argv)
    if args.function is None:
        build_parser().error("--function / -f is required (download or standardize).")
    return args
