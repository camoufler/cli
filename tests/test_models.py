"""Tests for model name validation."""

import pytest

from camoufler.models import ModelError, ensure_small, is_small_tag, parse_fq_name


@pytest.mark.parametrize(
    "model",
    [
        "qwen2.5:1.5b",
        "llama3.2:1b",
        "gemma2:2b",
        "phi3:mini",
        "smollm:360m",
        "qwen2.5:1.5b-instruct",
    ],
)
def test_accepts_small_models(model: str):
    assert ensure_small(model) == model


@pytest.mark.parametrize(
    "model",
    ["qwen2.5", "qwen2.5:7b", "llama3.1:8b", "mistral:latest", "foo:unknown"],
)
def test_rejects_invalid_models(model: str):
    with pytest.raises(ModelError):
        ensure_small(model)


def test_parse_fq_name():
    assert parse_fq_name("llama3.2:1b") == ("llama3.2", "1b")


def test_is_small_tag_aliases():
    assert is_small_tag("mini")
    assert is_small_tag("1.5b-instruct")
    assert not is_small_tag("7b")
