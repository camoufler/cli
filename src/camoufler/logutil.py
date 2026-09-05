"""Verbose logging setup."""

from __future__ import annotations

import logging
import sys


def setup_logging(verbose: int) -> logging.Logger:
    """Configure stderr logging from verbose level 0/1/2."""
    level = logging.WARNING
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    return logging.getLogger("camoufler")
