"""Tests for the application entry point."""

from unittest.mock import patch

from camoufler.app import main


def test_main_without_function_requires_tty(capsys):
    with patch("sys.stdin.isatty", return_value=False):
        assert main([]) == 1
    assert "terminal" in capsys.readouterr().err


@patch("camoufler.app.commands.cmd_list")
def test_main_list(mock_list):
    assert main(["-f", "list"]) == 0
    mock_list.assert_called_once()


@patch("camoufler.app.commands.cmd_set")
def test_main_set(mock_set):
    with patch("sys.stdin.isatty", return_value=False):
        assert main(["-f", "set", "-m", "qwen2.5:1.5b"]) == 0
    mock_set.assert_called_once()
    assert mock_set.call_args.kwargs["prompt"] is False
