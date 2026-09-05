"""Tests for command handlers."""

import argparse
from unittest.mock import patch

import pytest

from camoufler.commands import cmd_download, cmd_standardize, read_stdin
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
