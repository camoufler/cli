"""Persisted CLI settings (default model)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BUILTIN_DEFAULT_MODEL = "qwen2.5:1.5b"


def settings_path() -> Path:
    """Return ~/.camoufler/settings.json."""
    return Path.home() / ".camoufler" / "settings.json"


def load_settings() -> dict[str, Any]:
    """Load settings object; empty dict if missing or invalid."""
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_default_model(model: str) -> Path:
    """Write default_model and return the settings path."""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_settings()
    data["default_model"] = model
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def resolve_model(cli_model: str | None) -> str:
    """Prefer explicit -m, then saved default, then built-in 1.5B model."""
    if cli_model and cli_model.strip():
        return cli_model.strip()
    saved = load_settings().get("default_model")
    if isinstance(saved, str) and saved.strip():
        return saved.strip()
    return BUILTIN_DEFAULT_MODEL
