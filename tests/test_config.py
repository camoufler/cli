"""Tests for config loading."""

import json
from pathlib import Path

import pytest

from camoufler.config import ConfigError, cpu_options, load_config, resolve_config_path


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


def _write_cfg(path: Path, prompt: str) -> None:
    path.write_text(
        json.dumps({"system_prompt": prompt, "options": {}}),
        encoding="utf-8",
    )


def test_resolve_prefers_example2(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    _write_cfg(cfg_dir / "config.example.json", "one")
    _write_cfg(cfg_dir / "config.example2.json", "two")
    assert resolve_config_path(None).name == "config.example2.json"


def test_resolve_falls_back_to_example(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    _write_cfg(cfg_dir / "config.example.json", "one")
    assert resolve_config_path(None).name == "config.example.json"


def test_resolve_packaged_when_cwd_empty(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = resolve_config_path(None)
    assert path.name == "config.example.json"
    assert path.is_file()


def test_resolve_explicit_path(tmp_path: Path):
    target = tmp_path / "mine.json"
    assert resolve_config_path(str(target)) == target
