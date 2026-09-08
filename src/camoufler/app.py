"""Application entry point."""

from __future__ import annotations

import sys

from camoufler import args, commands, config, logutil
from camoufler.models import ModelError


def main(argv: list[str] | None = None) -> int:
    """Parse args, dispatch download or standardize, return exit code."""
    try:
        ns = args.parse_args(argv)
        logger = logutil.setup_logging(ns.verbose)
        if ns.function == "download":
            cfg = config.load_config(ns.config, required=False)
            if cfg:
                logger.debug("Config loaded (unused for download).")
            commands.cmd_download(ns, logger)
            return 0
        cfg = config.load_config(ns.config, required=True)
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
