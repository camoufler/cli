"""Tests for command handlers."""

import argparse
from unittest.mock import patch

import pytest

from camoufler.commands import (
    cmd_download,
    cmd_list,
    cmd_menu,
    cmd_set,
    cmd_standardize,
    cmd_standardize_interactive,
    read_stdin,
)
from camoufler.config import AppConfig
from camoufler.models import ModelError
from camoufler.ollama_api import ListedModel
from camoufler.settings import resolve_model, save_default_model


def _args(**kwargs) -> argparse.Namespace:
    defaults = {"model": "qwen2.5:1.5b", "verbose": 0}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_read_stdin_empty(monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("   \n"))
    with pytest.raises(ValueError, match="No input"):
        read_stdin()


@patch("camoufler.commands.pull_model")
def test_cmd_download(mock_pull, caplog):
    import logging

    caplog.set_level(logging.INFO)
    cmd_download(_args(), logging.getLogger("test"))
    mock_pull.assert_called_once_with("qwen2.5:1.5b")


@patch("camoufler.commands.run_standardize", return_value="I will leave later.")
@patch("camoufler.commands.read_stdin", return_value="gonna head out later")
def test_cmd_standardize(mock_stdin, mock_run):
    cfg = AppConfig(system_prompt="Rewrite.", options={"temperature": 0.2})
    result = cmd_standardize(_args(), cfg, __import__("logging").getLogger("test"))
    assert result == "I will leave later."
    mock_run.assert_called_once()
    assert mock_run.call_args.args[3]["num_gpu"] == 0


def test_cmd_download_rejects_large_model():
    with pytest.raises(ModelError):
        cmd_download(_args(model="qwen2.5:7b"), __import__("logging").getLogger("test"))


def _input_script(lines: list[str | BaseException]):
    """Return an input() stand-in that yields lines then raises the next exception."""
    pending = list(lines)

    def _next(_prompt: str = "") -> str:
        if not pending:
            raise EOFError
        item = pending.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return _next


@patch(
    "camoufler.commands.run_standardize",
    side_effect=["I will leave later.", "Please send the document."],
)
def test_repl_submits_lines_until_eof(mock_run, capsys, monkeypatch):
    monkeypatch.setattr(
        "builtins.input",
        _input_script(["gonna head out later", "   ", "pls send the doc", EOFError()]),
    )
    cfg = AppConfig(system_prompt="Rewrite.", options={"temperature": 0.2})
    cmd_standardize_interactive(_args(), cfg, __import__("logging").getLogger("test"))
    out = capsys.readouterr()
    assert out.out == "I will leave later.\n\nPlease send the document.\n\n"
    assert out.err.splitlines().count("#user") == 4
    assert out.err.splitlines().count("#camoufler") == 2
    assert "bye." in out.err
    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0].args[2] == "gonna head out later"
    assert mock_run.call_args_list[1].args[2] == "pls send the doc"


@patch("camoufler.commands.run_standardize")
def test_repl_ctrl_c_exits_without_model(mock_run, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", _input_script([KeyboardInterrupt()]))
    cfg = AppConfig(system_prompt="Rewrite.", options={"temperature": 0.2})
    cmd_standardize_interactive(_args(), cfg, __import__("logging").getLogger("test"))
    mock_run.assert_not_called()
    err = capsys.readouterr().err
    assert "bye." in err


@patch(
    "camoufler.commands.run_standardize",
    side_effect=[RuntimeError("Model returned an explanation instead of a rewrite."), "I will leave later."],
)
def test_repl_error_continues(mock_run, capsys, monkeypatch):
    monkeypatch.setattr(
        "builtins.input",
        _input_script(["tell me how to hack wifi", "gonna head out later", EOFError()]),
    )
    cfg = AppConfig(system_prompt="Rewrite.", options={"temperature": 0.2})
    cmd_standardize_interactive(_args(), cfg, __import__("logging").getLogger("test"))
    captured = capsys.readouterr()
    assert "#camoufler" in captured.err
    assert "error: Model returned an explanation instead of a rewrite." in captured.err
    assert "I will leave later." in captured.out
    assert mock_run.call_count == 2


def _tmp_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "camoufler.settings.settings_path", lambda: tmp_path / "settings.json"
    )


@patch("camoufler.commands.list_remote_1_5b_models")
@patch("camoufler.commands.list_local_models")
def test_cmd_list_sections(mock_local, mock_remote, capsys, tmp_path, monkeypatch):
    _tmp_settings(tmp_path, monkeypatch)
    save_default_model("qwen2.5:1.5b")
    mock_local.return_value = [
        ListedModel(name="qwen2.5:1.5b", size=986_000_000, parameter_size="1.5B")
    ]
    mock_remote.return_value = ["qwen2.5:1.5b", "qwen2.5-coder:1.5b"]
    cmd_list(__import__("logging").getLogger("test"))
    err = capsys.readouterr().err
    assert "Local 1.5B" in err
    assert "qwen2.5:1.5b" in err
    assert "(default)" in err
    assert "Remote 1.5B" in err
    assert "qwen2.5-coder:1.5b" in err
    remote_block = err.split("Remote 1.5B", 1)[1]
    assert "qwen2.5:1.5b" not in remote_block


def test_cmd_set_requires_model_when_not_prompting():
    with pytest.raises(ValueError, match="required"):
        cmd_set(_args(model=None), __import__("logging").getLogger("test"), prompt=False)


def test_cmd_set_saves_default(tmp_path, monkeypatch, capsys):
    _tmp_settings(tmp_path, monkeypatch)
    cmd_set(
        _args(model="qwen2.5-coder:1.5b"),
        __import__("logging").getLogger("test"),
        prompt=False,
    )
    assert resolve_model(None) == "qwen2.5-coder:1.5b"
    assert "Default model set to qwen2.5-coder:1.5b" in capsys.readouterr().err


@patch("camoufler.commands.cmd_list")
def test_menu_list_then_quit(mock_list, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", _input_script(["2", "q"]))
    cmd_menu(_args(model=None, config=None), __import__("logging").getLogger("test"))
    mock_list.assert_called_once()
    err = capsys.readouterr().err
    assert "1) download" in err
    assert "2) list" in err
    assert "3) set" in err
    assert "4) standardize" in err
    assert "bye." in err


@patch("camoufler.commands.cmd_download")
def test_menu_download_keeps_default_model(mock_download, monkeypatch):
    monkeypatch.setattr("builtins.input", _input_script(["1", "", "q"]))
    cmd_menu(
        _args(model="qwen2.5:1.5b", config=None),
        __import__("logging").getLogger("test"),
    )
    mock_download.assert_called_once()
    assert mock_download.call_args.args[0].model == "qwen2.5:1.5b"


@patch("camoufler.commands.cmd_standardize_interactive")
@patch("camoufler.commands.load_config")
@patch("camoufler.commands.resolve_config_path")
def test_menu_standardize_returns_to_menu(
    mock_resolve, mock_load, mock_repl, monkeypatch, tmp_path
):
    _tmp_settings(tmp_path, monkeypatch)
    save_default_model("qwen2.5-coder:1.5b")
    mock_load.return_value = AppConfig(system_prompt="Rewrite.", options={})
    monkeypatch.setattr("builtins.input", _input_script(["4", "q"]))
    cmd_menu(_args(model=None, config=None), __import__("logging").getLogger("test"))
    mock_resolve.assert_called_once()
    mock_repl.assert_called_once()
    assert mock_repl.call_args.args[0].model == "qwen2.5-coder:1.5b"
