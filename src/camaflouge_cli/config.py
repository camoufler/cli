"""JSON config loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when config file is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    system_prompt: str
    options: dict[str, Any]


def _require_config_path(path: str | None, required: bool) -> Path | None:
    if path:
        return Path(path)
    if required:
        raise ConfigError("--config / -c is required for standardize.")
    return None


def load_config(path: str | None, *, required: bool = False) -> AppConfig | None:
    """Load and validate JSON config; return None if path omitted."""
    cfg_path = _require_config_path(path, required)
    if cfg_path is None:
        return None
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return _parse_config(data, cfg_path)


def _parse_config(data: dict[str, Any], path: Path) -> AppConfig:
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be an object: {path}")
    prompt = data.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigError(f"Config must include non-empty 'system_prompt': {path}")
    options = data.get("options", {})
    if not isinstance(options, dict):
        raise ConfigError(f"'options' must be an object: {path}")
    return AppConfig(system_prompt=prompt.strip(), options=dict(options))


def cpu_options(options: dict[str, Any]) -> dict[str, Any]:
    """Merge config options and force CPU-only inference."""
    merged = dict(options)
    merged["num_gpu"] = 0
    return merged
