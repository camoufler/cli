"""Fully qualified model name validation."""

from __future__ import annotations

import re

SMALL_ALIASES = frozenset({"mini", "tiny"})
_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([bm])$")


class ModelError(ValueError):
    """Raised when a model name fails validation."""


def parse_fq_name(model: str) -> tuple[str, str]:
    """Split name:tag; reject bare names without a colon."""
    if ":" not in model:
        raise ModelError(
            f"Model must be fully qualified (name:tag), got '{model}'."
        )
    name, tag = model.rsplit(":", 1)
    if not name or not tag:
        raise ModelError(f"Invalid model reference '{model}'.")
    return name, tag


def _base_tag(tag: str) -> str:
    return tag.lower().split("-", 1)[0]


def is_small_tag(tag: str) -> bool:
    """Return True if the Ollama tag denotes a small CPU-runnable model."""
    base = _base_tag(tag)
    if base == "latest":
        return False
    if base in SMALL_ALIASES:
        return True
    match = _SIZE_RE.match(base)
    if not match:
        return False
    size, unit = float(match.group(1)), match.group(2)
    if unit == "m":
        return True
    return unit == "b" and size < 7


def is_1_5b_tag(tag: str) -> bool:
    """Return True if the tag denotes a 1.5B parameter variant."""
    return _base_tag(tag) == "1.5b"


def is_1_5b_model(model: str) -> bool:
    """Return True if model is a fully qualified 1.5B name:tag."""
    try:
        _, tag = parse_fq_name(model)
    except ModelError:
        return False
    return is_1_5b_tag(tag)


def ensure_small(model: str) -> str:
    """Validate FQ model name and ensure the tag is small enough for CPU."""
    _, tag = parse_fq_name(model)
    if not is_small_tag(tag):
        raise ModelError(
            f"Model '{model}' is not allowed. Use a small CPU tag "
            f"(under 7B, e.g. 1.5b, 1b, mini). Avoid 'latest' and 7b+."
        )
    return model


def ensure_1_5b(model: str) -> str:
    """Validate FQ model name and require a 1.5B tag."""
    _, tag = parse_fq_name(model)
    if not is_1_5b_tag(tag):
        raise ModelError(
            f"Model '{model}' is not a 1.5B tag. Use name:1.5b "
            f"(e.g. qwen2.5:1.5b)."
        )
    return model
