"""Tests for persisted default model settings."""

from pathlib import Path

from camoufler.settings import (
    BUILTIN_DEFAULT_MODEL,
    load_settings,
    resolve_model,
    save_default_model,
)


def _use_tmp_settings(tmp_path: Path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr("camoufler.settings.settings_path", lambda: path)
    return path


def test_resolve_prefers_cli_model(tmp_path: Path, monkeypatch):
    _use_tmp_settings(tmp_path, monkeypatch)
    save_default_model("qwen2.5-coder:1.5b")
    assert resolve_model("deepseek-r1:1.5b") == "deepseek-r1:1.5b"


def test_resolve_uses_saved_then_builtin(tmp_path: Path, monkeypatch):
    _use_tmp_settings(tmp_path, monkeypatch)
    assert resolve_model(None) == BUILTIN_DEFAULT_MODEL
    save_default_model("qwen2.5-coder:1.5b")
    assert resolve_model(None) == "qwen2.5-coder:1.5b"
    assert resolve_model("  ") == "qwen2.5-coder:1.5b"
    assert load_settings()["default_model"] == "qwen2.5-coder:1.5b"


def test_load_settings_invalid_file(tmp_path: Path, monkeypatch):
    path = _use_tmp_settings(tmp_path, monkeypatch)
    path.write_text("{not-json", encoding="utf-8")
    assert load_settings() == {}
    assert resolve_model(None) == BUILTIN_DEFAULT_MODEL
