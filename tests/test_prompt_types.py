"""Tests for heuristic prompt-type detection."""

from camoufler.prompt_types import PromptType, classify, classify_heuristic, parse_label


def test_slang_is_utterance():
    text = (
        "Yo, yesterday was lowkey wild. My main guy pulled up with some snacks "
        "that were totally bussin', and we just sat on the couch to veg out."
    )
    winner, _, tied = classify_heuristic(text)
    assert winner is PromptType.UTTERANCE
    assert tied is False


def test_gonna_is_utterance():
    winner, _, _ = classify_heuristic("gonna head out later")
    assert winner is PromptType.UTTERANCE


def test_factual_difference_question():
    winner, _, _ = classify_heuristic(
        "What is the difference between synchronous and asynchronous encryption?"
    )
    assert winner is PromptType.FACTUAL


def test_instructional_how_to():
    winner, _, _ = classify_heuristic(
        "Walk me through how to set up an automated backup pipeline using a cron job and rsync."
    )
    assert winner is PromptType.INSTRUCTIONAL


def test_creative_headlines():
    winner, _, _ = classify_heuristic(
        "Draft three short, compelling headlines for a product launch announcing "
        "an open-source analytics dashboard."
    )
    assert winner is PromptType.CREATIVE


def test_analytical_debug():
    winner, _, _ = classify_heuristic(
        "Here is a Python function throwing a KeyError; debug what went wrong "
        "and show the corrected code."
    )
    assert winner is PromptType.ANALYTICAL


def test_transformation_condense():
    winner, _, _ = classify_heuristic(
        "Condense this 4-page policy briefing into a five-bullet executive summary "
        "focusing on compliance risks."
    )
    assert winner is PromptType.TRANSFORMATION


def test_roleplay_interviewer():
    winner, _, _ = classify_heuristic(
        "Act as an interviewer conducting a technical screen for a systems "
        "architect role, and ask me three initial questions."
    )
    assert winner is PromptType.ROLEPLAY


def test_strategic_study_plan():
    winner, _, _ = classify_heuristic(
        "I need a 4-week study plan to prepare for a cloud certification, "
        "studying roughly 6 hours per week."
    )
    assert winner is PromptType.STRATEGIC


def test_parse_label():
    assert parse_label("factual") is PromptType.FACTUAL
    assert parse_label("Roleplay\n") is PromptType.ROLEPLAY
    assert parse_label("not-a-type") is None


def test_classify_uses_heuristic_without_chat():
    assert classify("gonna head out later") is PromptType.UTTERANCE


def test_classify_tie_break_uses_chat():
    text = "Bro how to fix my python ModuleNotFoundError for sacrebleu?"

    def _chat(_model: str, _text: str, _options: dict) -> str:
        return "analytical"

    winner, _, tied = classify_heuristic(text)
    assert tied is True
    assert (
        classify(text, model="qwen2.5:1.5b", options={}, classify_chat=_chat)
        is PromptType.ANALYTICAL
    )


def test_classify_invalid_chat_falls_back():
    text = "Bro how to fix my python ModuleNotFoundError for sacrebleu?"
    heuristic, _, tied = classify_heuristic(text)
    assert tied is True
    result = classify(
        text,
        model="qwen2.5:1.5b",
        options={},
        classify_chat=lambda *_a, **_k: "nope",
    )
    assert result is heuristic
