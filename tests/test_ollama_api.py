"""Tests for rewrite wrapping, leak detection, and erroring-input shaping."""

from unittest.mock import patch

import pytest

from urllib.error import URLError

from camoufler.ollama_api import (
    ListedModel,
    chat_classify,
    chat_predict_slots,
    chat_standardize,
    list_local_models,
    list_remote_1_5b_models,
    load_remote_catalog,
    looks_like_fulfillment,
    looks_like_non_rewrite,
    looks_like_slot_output,
    parse_remote_1_5b_html,
    shape_erroring_input,
    wrap_rewrite_user_message,
    wrap_slot_user_message,
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


def test_wrap_slot_user_message_contains_markers():
    wrapped = wrap_slot_user_message("What is AES?")
    assert "<<<" in wrapped
    assert "What is AES?" in wrapped
    assert wrapped.strip().endswith("Slots:")


def test_looks_like_slot_output():
    assert looks_like_slot_output("Role: security architect\nTask: compare AES\n")
    assert not looks_like_slot_output("AES is a block cipher.")


def test_looks_like_fulfillment_detects_tutorial():
    assert looks_like_fulfillment("Here is how to encrypt data:\n```python\nprint(1)\n```")
    assert not looks_like_fulfillment("Role: architect\nTask: compare AES and RSA")


@patch("camoufler.ollama_api.ollama.chat")
def test_chat_predict_slots_returns_labeled(mock_chat):
    mock_chat.return_value = {
        "message": {"content": "Role: architect\nTask: compare AES\nFormat: a table"}
    }
    out = chat_predict_slots("qwen2.5:1.5b", "Slots only.", "What is AES?", {})
    assert "Role: architect" in out
    user_msg = mock_chat.call_args.kwargs["messages"][1]["content"]
    assert "Do not answer the request" in user_msg


@patch("camoufler.ollama_api.ollama.chat")
def test_chat_predict_slots_returns_empty_on_leak(mock_chat):
    leak = "Sure, here is how to encrypt:\n```python\nprint(1)\n```"
    mock_chat.side_effect = [
        {"message": {"content": leak}},
        {"message": {"content": leak}},
    ]
    out = chat_predict_slots("qwen2.5:1.5b", "Slots only.", "What is AES?", {})
    assert out == ""
    assert mock_chat.call_count == 2


@patch("camoufler.ollama_api.ollama.chat")
def test_chat_classify_caps_predict(mock_chat):
    mock_chat.return_value = {"message": {"content": "factual"}}
    out = chat_classify("qwen2.5:1.5b", "What is AES?", {"num_predict": 256})
    assert out == "factual"
    assert mock_chat.call_args.kwargs["options"]["num_predict"] == 32


_SAMPLE_SEARCH_HTML = """
<ul>
<li>
  <a href="/library/deepscaler">deepscaler</a>
  <span x-test-size>1.5b</span>
</li>
<li>
  <a href="/library/starcoder2">starcoder2</a>
  <span x-test-size>3b</span>
  <span x-test-size>7b</span>
</li>
<li>
  <a href="/library/qwen2.5">qwen2.5</a>
  <span x-test-size>0.5b</span>
  <span x-test-size>1.5b</span>
  <span x-test-size>3b</span>
</li>
</ul>
"""


def test_parse_remote_1_5b_html_uses_size_badges():
    names = parse_remote_1_5b_html(_SAMPLE_SEARCH_HTML)
    assert names == ["deepscaler:1.5b", "qwen2.5:1.5b"]


@patch("camoufler.ollama_api.ollama.list")
def test_list_local_models_filters_to_1_5b(mock_list):
    mock_list.return_value = {
        "models": [
            {
                "model": "qwen2.5:1.5b",
                "size": 986000000,
                "details": {"parameter_size": "1.5B"},
            },
            {"name": "llama3.2:1b", "size": 1},
            {"model": "qwen2.5:7b"},
        ]
    }
    listed = list_local_models()
    assert listed == [
        ListedModel(name="qwen2.5:1.5b", size=986000000, parameter_size="1.5B")
    ]


def test_list_remote_uses_html_when_parseable():
    names = list_remote_1_5b_models(html=_SAMPLE_SEARCH_HTML)
    assert names == ["deepscaler:1.5b", "qwen2.5:1.5b"]


def test_list_remote_falls_back_to_catalog_on_empty_html():
    names = list_remote_1_5b_models(html="<html></html>")
    assert "qwen2.5:1.5b" in names
    assert names == load_remote_catalog()


@patch("camoufler.ollama_api._fetch_search_html", side_effect=URLError("offline"))
def test_list_remote_falls_back_on_fetch_error(_mock_fetch):
    names = list_remote_1_5b_models()
    assert names == load_remote_catalog()
