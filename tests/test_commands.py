"""Tests for command handlers."""

import argparse
from unittest.mock import patch

import pytest

from camoufler.commands import (
    cmd_download,
    cmd_standardize,
    cmd_standardize_interactive,
    iter_interactive_inputs,
    read_stdin,
)
from camoufler.config import AppConfig
from camoufler.models import ModelError


class _InteractiveStdin:
    """Mock stdin that simulates readline() with optional KeyboardInterrupt."""

    def __init__(self, actions: list[str | type[KeyboardInterrupt]]) -> None:
        self._actions = list(actions)
        self._index = 0

    def readline(self) -> str:
        if self._index >= len(self._actions):
            return ""
        action = self._actions[self._index]
        self._index += 1
        if action is KeyboardInterrupt:
            raise KeyboardInterrupt
        if action == "":
            return ""
        return action if action.endswith("\n") else f"{action}\n"


def _collect_interactive(actions: list[str | type[KeyboardInterrupt]]) -> list[str]:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("sys.stdin", _InteractiveStdin(actions))
    try:
        return list(iter_interactive_inputs())
    finally:
        monkeypatch.undo()


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


def test_iter_interactive_inputs_ctrl_c_submits():
    chunks = _collect_interactive(["gonna head out later", KeyboardInterrupt, ""])
    assert chunks == ["gonna head out later"]


@patch(
    "camoufler.commands._partial_line",
    return_value="ok i need a bike now immediately",
)
def test_iter_interactive_inputs_ctrl_c_submits_unfinished_line(mock_partial):
    chunks = _collect_interactive([KeyboardInterrupt, ""])
    assert chunks == ["ok i need a bike now immediately"]
    mock_partial.assert_called_once()


@patch("camoufler.commands._partial_line", return_value="partial")
def test_iter_interactive_inputs_ctrl_c_combines_buffer_and_partial(mock_partial):
    chunks = _collect_interactive(["line one", KeyboardInterrupt, ""])
    assert chunks == ["line one\npartial"]
    mock_partial.assert_called_once()


def test_iter_interactive_inputs_multiple_rounds():
    chunks = _collect_interactive(
        [
            "first",
            KeyboardInterrupt,
            "second",
            "line two",
            KeyboardInterrupt,
            "",
        ]
    )
    assert chunks == ["first", "second\nline two"]


def test_iter_interactive_inputs_eof_exits():
    chunks = _collect_interactive(["unused text", ""])
    assert chunks == []


def test_iter_interactive_inputs_empty_ctrl_c_ignored():
    chunks = _collect_interactive([KeyboardInterrupt, "hello", KeyboardInterrupt, ""])
    assert chunks == ["hello"]


@patch(
    "camoufler.commands.chat_standardize",
    side_effect=["First result.", "Second result."],
)
@patch(
    "camoufler.commands.iter_interactive_inputs",
    return_value=iter(["first input", "second input"]),
)
def test_cmd_standardize_interactive(mock_inputs, mock_chat, capsys):
    cfg = AppConfig(system_prompt="Rewrite.", options={"temperature": 0.2})
    cmd_standardize_interactive(_args(), cfg, __import__("logging").getLogger("test"))
    out = capsys.readouterr().out
    assert out == "First result.\nSecond result.\n"
    assert mock_chat.call_count == 2
    assert mock_chat.call_args_list[0].args[2] == "first input"
    assert mock_chat.call_args_list[1].args[2] == "second input"
