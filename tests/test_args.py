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


def test_function_required():
    with __import__("pytest").raises(SystemExit):
        parse_args([])
