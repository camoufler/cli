"""Tests for RACE gap detection, model merge, and paragraph assembly."""

from camoufler.race import (
    assemble_race,
    complete_from_prediction,
    merge_predicted_slots,
    missing_slots,
    needs_prediction,
    seed_answers,
)


BLOG = "Write me a blog post about AI"


def test_blog_prompt_missing_role_context_expectations():
    assert missing_slots(BLOG) == ["role", "context", "expectations"]
    assert needs_prediction(BLOG) is True


def test_labeled_race_prompt_is_complete():
    text = (
        "Role: You are a technical content writer for developers new to ML.\n"
        "Action: Create an outline for a blog post explaining gradient descent.\n"
        "Context: Audience is mid-level web developers without much calculus.\n"
        "Expectations: Structured outline, conversational but accurate."
    )
    assert missing_slots(text) == []
    assert needs_prediction(text) is False


def test_heuristic_complete_without_labels():
    text = (
        "You are a copywriter specializing in SaaS landing pages. "
        "Draft a follow-up email that politely requests a status update. "
        "I'm building a React app for a fintech startup. "
        "Keep it under 150 words."
    )
    assert missing_slots(text) == []
    # Role and Action can be seeded; Context/Expectations still need values.
    assert needs_prediction(text) is True


def test_merge_rejects_glued_action_as_role():
    original = "write a blog post about AI application"
    predicted = (
        "Role: Action: Write\n"
        "Action: Write\n"
        "Context: a blog post about AI application\n"
        "Expectations: a detailed and informative blog post about the application of AI"
    )
    merged = merge_predicted_slots(original, predicted)
    assert "role" not in merged
    assert merged["action"] == original
    assert merged["context"] == "a general audience"
    paragraph = complete_from_prediction(original, predicted)
    assert "You are an Action" not in paragraph
    assert "You are a technical writer." not in paragraph
    assert "for a blog post about AI application" not in paragraph
    assert "Write a blog post about AI application for a general audience." in paragraph


def test_merge_keeps_user_action_and_fills_gaps():
    predicted = (
        "Role: technical writer\n"
        "Action: ignore this\n"
        "Context: general readers who are novice in AI\n"
        "Expectations: under 40 words, simple, genius"
    )
    merged = merge_predicted_slots(BLOG, predicted)
    assert merged["action"] == BLOG
    assert "role" not in merged
    assert merged["context"] == "general readers who are novice in AI"
    assert merged["expectations"] == "under 40 words, simple, genius"


def test_complete_from_prediction_is_one_paragraph():
    predicted = (
        "Role: technical writer\n"
        "Action: write a blog about AI\n"
        "Context: general readers who are novice in AI\n"
        "Expectations: under 40 words, simple, genius"
    )
    paragraph = complete_from_prediction(BLOG, predicted)
    assert "\n" not in paragraph
    assert paragraph == (
        "Write a blog post about AI "
        "for general readers who are novice in AI. "
        "Keep it under 40 words, simple, genius."
    )


def test_assemble_race_is_one_paragraph():
    paragraph = assemble_race(
        {
            "role": "technical writer",
            "action": "write me a blog about AI",
            "context": "general readers who are novice in AI",
            "expectations": "under 40 words , simple , genius",
        }
    )
    assert "\n" not in paragraph
    assert paragraph == (
        "You are a technical writer. Write a blog about AI "
        "for general readers who are novice in AI. "
        "Keep it under 40 words, simple, genius."
    )


def test_complete_from_prediction_uses_labeled_without_model():
    text = (
        "Role: You are a copywriter.\n"
        "Action: Draft a follow-up email.\n"
        "Context: Existing customer, 3-year relationship.\n"
        "Expectations: Under 150 words, no guilt."
    )
    out = complete_from_prediction(text)
    assert "Role:" not in out
    assert out == (
        "You are a copywriter. Draft a follow-up email "
        "for existing customer, 3-year relationship. "
        "Keep it under 150 words, no guilt."
    )


def test_seed_action_from_original():
    answers = seed_answers(BLOG)
    assert answers["action"] == BLOG
    assert "role" not in answers


PROOFREAD = (
    'Proof read this text "Yo, yesterday was lowkey wild. My main guy pulled up '
    "with some snacks that were totally bussin'.\""
)
SLANG = (
    "Yo, yesterday was lowkey wild. My main guy pulled up with some snacks "
    "that were totally bussin', and we just sat on the couch to veg out."
)


def test_rejects_none_and_narrator_for_proofread():
    predicted = (
        "Role: None\n"
        "Action: None\n"
        "Context: a text describing a social interaction\n"
        "Expectations: the text is to be corrected for spelling, grammar, and style"
    )
    paragraph = complete_from_prediction(PROOFREAD, predicted)
    assert "You are a None" not in paragraph
    assert "You are a copy editor." in paragraph
    assert "Proofread the quoted text into standard English" in paragraph
    assert "informal spoken English" in paragraph
    assert "Keep the meaning and similar length." in paragraph


def test_rejects_narrator_summary_for_slang_utterance():
    predicted = (
        "Role: Narrator\n"
        "Action: None\n"
        "Context: a conversation about a past event\n"
        "Expectations: provide a summary of the conversation"
    )
    paragraph = complete_from_prediction(SLANG, predicted)
    assert "You are a Narrator" not in paragraph
    assert "None for" not in paragraph
    assert paragraph.startswith("You are a copy editor.")
    assert "Rewrite the text into standard English" in paragraph
    assert "informal spoken English" in paragraph


def test_user_voice_email_rejects_support_role():
    original = (
        "I wanna write an email about my ninja blender, it stop working and I m pissed of."
    )
    predicted = (
        "Role: customer support representative\n"
        "Action: apologize and offer a replacement\n"
        "Context: Ninja blender customers\n"
        "Expectations: apologize, offer a solution, provide a timeframe"
    )
    merged = merge_predicted_slots(original, predicted)
    assert "role" not in merged
    assert "pissed" in merged["action"].lower() or "blender" in merged["action"].lower()
    paragraph = complete_from_prediction(original, predicted)
    assert "You are" not in paragraph
    assert "customer support" not in paragraph.lower()
    assert "pissed" in paragraph.lower() or "blender" in paragraph.lower()

