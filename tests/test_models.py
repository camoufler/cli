"""Tests for model name validation."""

import pytest

from camoufler.models import (
    ModelError,
    ensure_1_5b,
    ensure_small,
    is_1_5b_model,
    is_1_5b_tag,
    is_small_tag,
    parse_fq_name,
)


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


def test_ensure_1_5b_accepts_instruct_suffix():
    assert ensure_1_5b("qwen2.5:1.5b") == "qwen2.5:1.5b"
    assert ensure_1_5b("qwen2.5:1.5b-instruct") == "qwen2.5:1.5b-instruct"
    assert is_1_5b_tag("1.5b-instruct")
    assert is_1_5b_model("user/qwen:1.5b")


@pytest.mark.parametrize("model", ["llama3.2:1b", "qwen2.5:7b", "phi3:mini", "qwen2.5"])
def test_ensure_1_5b_rejects_other_sizes(model: str):
    with pytest.raises(ModelError):
        ensure_1_5b(model)
