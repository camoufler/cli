"""Tests for CLI argument parsing."""

from camoufler import __version__
from camoufler.args import build_parser, parse_args


def test_version_flag(capsys):
    with __import__("pytest").raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert f"camoufler {__version__}" in out


def test_parse_download_args():
    ns = parse_args(["-f", "download", "-m", "qwen2.5:1.5b", "--verbose", "1"])
    assert ns.function == "download"
    assert ns.model == "qwen2.5:1.5b"
    assert ns.verbose == 1


def test_parse_empty_args_opens_menu_path():
    ns = parse_args([])
    assert ns.function is None
    assert ns.model is None


def test_parse_list_and_set():
    assert parse_args(["-f", "list"]).function == "list"
    ns = parse_args(["-f", "set", "-m", "qwen2.5-coder:1.5b"])
    assert ns.function == "set"
    assert ns.model == "qwen2.5-coder:1.5b"
