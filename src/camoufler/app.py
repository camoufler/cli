"""Application entry point."""

from __future__ import annotations

import sys

from camoufler import args, commands, config, logutil
from camoufler.models import ModelError


def main(argv: list[str] | None = None) -> int:
    """Parse args, dispatch a command or the TTY menu, return exit code."""
    try:
        ns = args.parse_args(argv)
        logger = logutil.setup_logging(ns.verbose)
        if ns.function is None:
            if not sys.stdin.isatty():
                raise ValueError(
                    "Run camoufler on a terminal for the menu, or pass -f "
                    "(download, list, set, standardize)."
                )
            commands.cmd_menu(ns, logger)
            return 0
        if ns.function == "download":
            cfg = config.load_config(ns.config, required=False)
            if cfg:
                logger.debug("Config loaded (unused for download).")
            commands.cmd_download(ns, logger)
            return 0
        if ns.function == "list":
            commands.cmd_list(logger)
            return 0
        if ns.function == "set":
            commands.cmd_set(ns, logger, prompt=sys.stdin.isatty())
            return 0
        cfg_path = config.resolve_config_path(ns.config)
        cfg = config.load_config(str(cfg_path), required=True)
        assert cfg is not None
        if sys.stdin.isatty():
            commands.cmd_standardize_interactive(ns, cfg, logger)
            return 0
        result = commands.cmd_standardize(ns, cfg, logger)
        print(result)
        return 0
    except (ModelError, config.ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — top-level CLI guard
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
