"""Tests for command handlers."""

import argparse
from unittest.mock import patch

import pytest

from camoufler.commands import (
    cmd_download,
    cmd_standardize,
    cmd_standardize_interactive,
    read_stdin,
)
from camoufler.config import AppConfig
from camoufler.models import ModelError


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


@patch("camoufler.commands.chat_standardize", return_value="I will leave later.")
@patch("camoufler.commands.read_stdin", return_value="gonna head out later")
def test_cmd_standardize(mock_stdin, mock_chat):
    cfg = AppConfig(system_prompt="Rewrite.", options={"temperature": 0.2})
    result = cmd_standardize(_args(), cfg, __import__("logging").getLogger("test"))
    assert result == "I will leave later."
    mock_chat.assert_called_once()
    assert mock_chat.call_args.args[3]["num_gpu"] == 0


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
    "camoufler.commands.chat_standardize",
    side_effect=["I will leave later.", "Please send the document."],
)
def test_repl_submits_lines_until_eof(mock_chat, capsys, monkeypatch):
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
    assert mock_chat.call_count == 2
    assert mock_chat.call_args_list[0].args[2] == "gonna head out later"
    assert mock_chat.call_args_list[1].args[2] == "pls send the doc"


@patch("camoufler.commands.chat_standardize")
def test_repl_ctrl_c_exits_without_model(mock_chat, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", _input_script([KeyboardInterrupt()]))
    cfg = AppConfig(system_prompt="Rewrite.", options={"temperature": 0.2})
    cmd_standardize_interactive(_args(), cfg, __import__("logging").getLogger("test"))
    mock_chat.assert_not_called()
    err = capsys.readouterr().err
    assert "bye." in err


@patch(
    "camoufler.commands.chat_standardize",
    side_effect=[RuntimeError("Model returned an explanation instead of a rewrite."), "I will leave later."],
)
def test_repl_error_continues(mock_chat, capsys, monkeypatch):
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
    assert mock_chat.call_count == 2
