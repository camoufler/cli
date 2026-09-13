"""download, list, set, and standardize command handlers."""

from __future__ import annotations

import argparse
import logging
import sys

from camoufler import __version__
from camoufler.config import AppConfig, cpu_options, load_config, resolve_config_path
from camoufler.models import ensure_1_5b, ensure_small, is_1_5b_model
from camoufler.ollama_api import (
    ListedModel,
    list_local_models,
    list_remote_1_5b_models,
    pull_model,
)
from camoufler.pipeline import run_standardize
from camoufler.settings import BUILTIN_DEFAULT_MODEL, resolve_model, save_default_model


def read_stdin() -> str:
    """Read all stdin; error if empty."""
    text = sys.stdin.read()
    if not text.strip():
        raise ValueError("No input text on stdin.")
    return text.strip()


def _resolved_small(args: argparse.Namespace) -> str:
    return ensure_small(resolve_model(getattr(args, "model", None)))


def _with_model(args: argparse.Namespace, model: str) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.model = model
    return ns


def _format_bytes(size: int) -> str:
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.1f} GB"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.0f} MB"
    return f"{size} B"


def _local_extra(item: ListedModel) -> str:
    parts: list[str] = []
    if item.parameter_size:
        parts.append(item.parameter_size)
    if item.size is not None:
        parts.append(_format_bytes(item.size))
    return f"  {' '.join(parts)}" if parts else ""


def cmd_download(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Download the specified small Ollama model."""
    model = _resolved_small(args)
    logger.info("Downloading %s", model)
    pull_model(model)
    logger.info("Download complete: %s", model)


def cmd_list(logger: logging.Logger) -> None:
    """Print local and remote 1.5B models."""
    default = resolve_model(None)
    local: list[ListedModel] = []
    local_error: Exception | None = None
    try:
        local = list_local_models()
    except Exception as exc:  # noqa: BLE001 — still show remote
        local_error = exc
        logger.info("Local list failed: %s", exc)

    local_names = {item.name for item in local}
    remote: list[str] = []
    remote_note: str | None = None
    try:
        remote = [name for name in list_remote_1_5b_models() if name not in local_names]
    except Exception as exc:  # noqa: BLE001 — keep the command useful
        remote_note = str(exc)
        logger.info("Remote list failed: %s", exc)

    print("Local 1.5B", file=sys.stderr)
    if local_error is not None:
        print(f"  (could not list local models: {local_error})", file=sys.stderr)
    elif not local:
        print("  (none)", file=sys.stderr)
    else:
        for item in local:
            marker = "*" if item.name == default else " "
            suffix = "  (default)" if item.name == default else ""
            print(
                f" {marker} {item.name}{_local_extra(item)}{suffix}",
                file=sys.stderr,
            )

    print("\nRemote 1.5B", file=sys.stderr)
    if remote_note is not None:
        print(f"  (could not list remote models: {remote_note})", file=sys.stderr)
    elif not remote:
        print("  (none)", file=sys.stderr)
    else:
        for name in remote:
            marker = "*" if name == default else " "
            suffix = "  (default)" if name == default else ""
            print(f" {marker} {name}{suffix}", file=sys.stderr)


def cmd_set(args: argparse.Namespace, logger: logging.Logger, *, prompt: bool) -> None:
    """Save the default 1.5B model used by standardize."""
    raw = getattr(args, "model", None)
    if isinstance(raw, str):
        raw = raw.strip() or None
    if not raw and prompt:
        prefill = resolve_model(None)
        if not is_1_5b_model(prefill):
            prefill = BUILTIN_DEFAULT_MODEL
        print(f"Default model [{prefill}]:", file=sys.stderr)
        sys.stderr.flush()
        entered = input().strip()
        raw = entered or prefill
    if not raw:
        raise ValueError("Model is required for set. Pass -m name:tag.")
    model = ensure_1_5b(raw)
    save_default_model(model)
    logger.info("Default model set to %s", model)
    print(f"Default model set to {model}", file=sys.stderr)


def cmd_standardize(
    args: argparse.Namespace,
    config: AppConfig,
    logger: logging.Logger,
) -> str:
    """Detect prompt type, then rewrite or expand stdin text."""
    model = _resolved_small(args)
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
    model = _resolved_small(args)
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


def _prompt_download_model(args: argparse.Namespace) -> str:
    default = resolve_model(getattr(args, "model", None))
    print(f"Model [{default}]:", file=sys.stderr)
    sys.stderr.flush()
    entered = input().strip()
    return entered or default


def cmd_menu(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Interactive download / list / set / standardize loop."""
    while True:
        default = resolve_model(getattr(args, "model", None))
        print(
            f"camoufler {__version__}  default {default}\n"
            "  1) download\n"
            "  2) list\n"
            "  3) set\n"
            "  4) standardize\n"
            "  q) quit\n",
            file=sys.stderr,
        )
        print("Choice:", file=sys.stderr)
        sys.stderr.flush()
        try:
            choice = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.", file=sys.stderr)
            return
        if choice in {"q", "quit", "exit"}:
            print("bye.", file=sys.stderr)
            return
        if choice in {"1", "download", "d"}:
            try:
                model = _prompt_download_model(args)
                cmd_download(_with_model(args, model), logger)
            except (EOFError, KeyboardInterrupt):
                print("\nbye.", file=sys.stderr)
                return
            except Exception as exc:  # noqa: BLE001 — keep the menu alive
                print(f"error: {exc}", file=sys.stderr)
            continue
        if choice in {"2", "list", "l"}:
            try:
                cmd_list(logger)
            except Exception as exc:  # noqa: BLE001 — keep the menu alive
                print(f"error: {exc}", file=sys.stderr)
            print(file=sys.stderr)
            continue
        if choice in {"3", "set"}:
            try:
                cmd_set(_with_model(args, None), logger, prompt=True)
            except (EOFError, KeyboardInterrupt):
                print("\nbye.", file=sys.stderr)
                return
            except Exception as exc:  # noqa: BLE001 — keep the menu alive
                print(f"error: {exc}", file=sys.stderr)
            continue
        if choice in {"4", "standardize", "s"}:
            try:
                cfg = load_config(str(resolve_config_path(getattr(args, "config", None))))
                assert cfg is not None
                cmd_standardize_interactive(_with_model(args, default), cfg, logger)
            except Exception as exc:  # noqa: BLE001 — keep the menu alive
                print(f"error: {exc}", file=sys.stderr)
            continue
        print("Unknown choice. Enter 1-4 or q.\n", file=sys.stderr)
