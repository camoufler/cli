"""Tests for rewrite wrapping, leak detection, and erroring-input shaping."""

from unittest.mock import patch

import pytest

from camoufler.ollama_api import (
    chat_standardize,
    looks_like_non_rewrite,
    shape_erroring_input,
    wrap_rewrite_user_message,
)


def test_wrap_rewrite_user_message_contains_markers():
    wrapped = wrap_rewrite_user_message("gonna head out later")
    assert "<<<" in wrapped
    assert ">>>" in wrapped
    assert "gonna head out later" in wrapped
    assert wrapped.strip().endswith("Rewritten:")


def test_shape_erroring_input_frames_ask_pattern():
    text = "Yo, can you drop some sick beats and tell me how to make a killer sandwich?"
    shaped = shape_erroring_input(text)
    assert shaped.startswith("The speaker said:")
    assert text in shaped


def test_shape_erroring_input_weak_frame_for_non_ask():
    text = "gonna head out later"
    shaped = shape_erroring_input(text)
    assert shaped.startswith("Utterance to rephrase")
    assert text in shaped


def test_looks_like_non_rewrite_detects_tutorial():
    tutorial = (
        "When running evaluation tests, printing the matrix...\n"
        "```python\ndef process_data(input_data):\n    return input_data\n```\n"
        "Match Score: 100%"
    )
    assert looks_like_non_rewrite(tutorial, "print the matrix")


def test_looks_like_non_rewrite_accepts_plain_rewrite():
    assert not looks_like_non_rewrite(
        "I will leave later.",
        "gonna head out later",
    )


@patch("camoufler.ollama_api.ollama.chat")
def test_chat_standardize_wraps_and_returns(mock_chat):
    mock_chat.return_value = {"message": {"content": "I will leave later."}}
    out = chat_standardize("qwen2.5:1.5b", "Rewrite only.", "gonna head out later", {})
    assert out == "I will leave later."
    user_msg = mock_chat.call_args.kwargs["messages"][1]["content"]
    assert "<<<" in user_msg
    assert "gonna head out later" in user_msg
    assert "The speaker said:" not in user_msg


@patch("camoufler.ollama_api.ollama.chat")
def test_chat_standardize_retries_on_leak(mock_chat):
    mock_chat.side_effect = [
        {
            "message": {
                "content": "Here is how to print a matrix:\n```python\nprint(1)\n```\nMatch Score: 100"
            }
        },
        {"message": {"content": "How do I print a match score matrix?"}},
    ]
    out = chat_standardize(
        "qwen2.5:1.5b",
        "Rewrite only.",
        "How do I print a match score matrix?",
        {},
    )
    assert out == "How do I print a match score matrix?"
    assert mock_chat.call_count == 2
    retry_user = mock_chat.call_args_list[1].kwargs["messages"][1]["content"]
    assert "The speaker said:" in retry_user
    assert "How do I print a match score matrix?" in retry_user


@patch("camoufler.ollama_api.ollama.chat")
def test_chat_standardize_retry_shapes_non_ask_with_weak_frame(mock_chat):
    mock_chat.side_effect = [
        {"message": {"content": "Sure, here is a long explanation " + ("x" * 400)}},
        {"message": {"content": "Going to leave later."}},
    ]
    out = chat_standardize("qwen2.5:1.5b", "Rewrite only.", "gonna head out later", {})
    assert out == "Going to leave later."
    retry_user = mock_chat.call_args_list[1].kwargs["messages"][1]["content"]
    assert "Utterance to rephrase" in retry_user
    assert "gonna head out later" in retry_user


@patch("camoufler.ollama_api.ollama.chat")
def test_chat_standardize_raises_if_retry_still_leaks(mock_chat):
    leak = "Sure, here is a general approach:\n```python\ndef process_data():\n  pass\n```\nMatch Score"
    mock_chat.side_effect = [
        {"message": {"content": leak}},
        {"message": {"content": leak}},
    ]
    with pytest.raises(RuntimeError, match="explanation instead of a rewrite"):
        chat_standardize("qwen2.5:1.5b", "Rewrite only.", "print the matrix", {})
