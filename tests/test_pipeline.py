"""Tests for detect → expand/rewrite → anonymize pipeline."""

from unittest.mock import patch

from camoufler.config import AppConfig
from camoufler.pipeline import run_standardize


CFG = AppConfig(system_prompt="Rewrite only.", options={"temperature": 0.0})


@patch("camoufler.pipeline.chat_standardize", return_value="I will leave later.")
def test_utterance_rewrites_then_anonymizes(mock_chat):
    out = run_standardize("qwen2.5:1.5b", CFG, "gonna head out later", {})
    assert out == "I will leave later."
    mock_chat.assert_called_once()


@patch(
    "camoufler.pipeline.chat_standardize",
    return_value="Call me at 555-0107 tomorrow.",
)
def test_utterance_anonymizes_rewrite(mock_chat):
    out = run_standardize("qwen2.5:1.5b", CFG, "hit me up at 555-0107 tmrw", {})
    assert "555-0107" not in out
    assert "[PHONE]" in out
    mock_chat.assert_called_once()


@patch("camoufler.pipeline.chat_standardize", side_effect=lambda _m, _s, text, _o: text)
@patch("camoufler.pipeline.chat_predict_slots")
def test_factual_expands_to_paragraph(mock_slots, mock_rewrite):
    mock_slots.return_value = (
        "Role: enterprise information security architect\n"
        "Task: differentiate symmetric vs. asymmetric encryption\n"
        "Format: a 3-row comparison table followed by a 2-sentence summary\n"
    )
    ask = "What is the difference between synchronous and asynchronous encryption?"
    out = run_standardize("qwen2.5:1.5b", CFG, ask, {})
    assert "\n" not in out
    assert "You are an enterprise information security architect." in out
    assert "```" not in out
    mock_rewrite.assert_called_once()
    mock_slots.assert_called_once()
    assert mock_slots.call_args.args[2] == ask


@patch("camoufler.pipeline.chat_standardize", side_effect=lambda _m, _s, text, _o: text)
@patch("camoufler.pipeline.chat_predict_slots", return_value="")
def test_empty_prediction_uses_defaults(mock_slots, mock_rewrite):
    ask = "Walk me through how to set up an automated backup using cron and rsync."
    out = run_standardize("qwen2.5:1.5b", CFG, ask, {})
    assert "Provide numbered setup steps" in out
    assert "You are" not in out or "Configure" in out or "Walk me" in out
    mock_slots.assert_called_once()


@patch("camoufler.pipeline.chat_standardize", side_effect=lambda _m, _s, text, _o: text)
@patch("camoufler.pipeline.chat_predict_slots")
def test_expansion_anonymizes_paragraph(mock_slots, mock_rewrite):
    mock_slots.return_value = (
        "Role: security architect\n"
        "Task: compare encryption for asmith@initech.corp\n"
        "Format: a short table\n"
    )
    ask = "What is the difference between AES and RSA for asmith@initech.corp?"
    out = run_standardize("qwen2.5:1.5b", CFG, ask, {})
    assert "asmith@initech.corp" not in out
    assert "[EMAIL]" in out


@patch(
    "camoufler.pipeline.chat_standardize",
    return_value="Debug the Python function that raises an error and show the corrected code.",
)
@patch(
    "camoufler.pipeline.chat_predict_slots",
    return_value=(
        "Role: software engineer\n"
        "Action: debug the Python function\n"
        "Context: a general audience\n"
        "Expectations: clear and concise\n"
    ),
)
def test_prompt_path_rewrites_before_expand(mock_slots, mock_rewrite):
    ask = (
        "This is python function throwing error, can you debug what went wrong "
        "and show the right code"
    )
    out = run_standardize("qwen2.5:1.5b", CFG, ask, {})
    assert "This is python function throwing error" not in out
    assert "Debug the Python function that raises an error" in out
    assert "You are a software engineer." in out
    mock_rewrite.assert_called_once()
    cleaned = mock_slots.call_args.args[2]
    assert cleaned.startswith("Debug the Python function")
