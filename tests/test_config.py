"""Tests for config loading."""

import json
from pathlib import Path

import pytest

from camaflouge_cli.config import ConfigError, cpu_options, load_config


def test_load_valid_config(tmp_path: Path):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "system_prompt": "Be concise.",
                "options": {"temperature": 0.1},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(str(path), required=True)
    assert cfg is not None
    assert cfg.system_prompt == "Be concise."
    assert cfg.options["temperature"] == 0.1


def test_config_required_for_standardize():
    with pytest.raises(ConfigError, match="required"):
        load_config(None, required=True)


def test_cpu_options_forces_num_gpu():
    opts = cpu_options({"temperature": 0.5, "num_gpu": 99})
    assert opts["num_gpu"] == 0
    assert opts["temperature"] == 0.5
