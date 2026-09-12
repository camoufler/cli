"""Tests for framework slot merge and paragraph assembly."""

from camoufler.frameworks import (
    CREATE,
    GRADE,
    RTF,
    TAG,
    TRAC,
    COAST,
    assemble_create,
    assemble_grade,
    assemble_rtf,
    assemble_tag,
    assemble_trac,
    assemble_coast,
    complete_from_prediction,
    merge_predicted_slots,
    needs_prediction,
)


def test_assemble_rtf_one_paragraph():
    paragraph = assemble_rtf(
        {
            "role": "enterprise information security architect",
            "task": "Differentiate symmetric vs. asymmetric encryption",
            "format": "a 3-row comparison table followed by a 2-sentence summary",
        }
    )
    assert "\n" not in paragraph
    assert paragraph.startswith("You are an enterprise information security architect.")
    assert "Differentiate symmetric vs. asymmetric encryption." in paragraph
    assert "Output a 3-row comparison table" in paragraph


def test_assemble_tag_one_paragraph():
    paragraph = assemble_tag(
        {
            "task": "Configure an automated incremental server backup using rsync and cron",
            "action": "Provide numbered setup steps including SSH keys",
            "goal": "each step includes a bash command to verify success",
        }
    )
    assert "\n" not in paragraph
    assert paragraph.startswith("Configure an automated incremental")
    assert "Ensure each step includes a bash command" in paragraph


def test_assemble_create_one_paragraph():
    paragraph = assemble_create(
        {
            "character": "direct-response B2B copywriter",
            "request": "Draft 3 product launch headlines",
            "examples": "crisp, technical, and benefit-driven",
            "adjustments": "No marketing jargon or hype words",
            "type": "bulleted list with sub-bullets",
            "extras": "Keep each headline under 10 words",
        }
    )
    assert "\n" not in paragraph
    assert paragraph.startswith("You are a direct-response B2B copywriter.")


def test_assemble_trac_one_paragraph():
    paragraph = assemble_trac(
        {
            "task": "Condense the briefing into an executive summary",
            "role": "strategic management consultant",
            "audience": "C-suite executives with limited technical time",
            "constraints": "Maximum 5 bullet points. Do not add facts.",
        }
    )
    assert "You are a strategic management consultant." in paragraph
    assert "for C-suite executives" in paragraph


def test_assemble_coast_one_paragraph():
    paragraph = assemble_coast(
        {
            "context": "Technical screening round for a Senior Cloud Infrastructure Lead",
            "objective": "Assess candidate depth in distributed system resiliency",
            "actor": "principal infrastructure architect",
            "scenario": "Ask one scenario-based question at a time",
            "tone": "professional, rigorous, and probing",
        }
    )
    assert paragraph.startswith("You are a principal infrastructure architect.")
    assert "Keep the tone professional" in paragraph


def test_assemble_grade_one_paragraph():
    paragraph = assemble_grade(
        {
            "goal": "Prepare for the AWS Certified Solutions Architect - Associate exam",
            "role": "cloud certification mentor",
            "assumptions": "4-week timeline, 6 hours/week",
            "deliverables": "Weekly study breakdown and practice exams",
            "evaluation": "scoring 85%+ on official timed practice assessments",
        }
    )
    assert paragraph.startswith("You are a cloud certification mentor.")
    assert "Success: scoring 85%+" in paragraph


def test_rtf_complete_from_labeled_without_model():
    text = (
        "Role: enterprise information security architect\n"
        "Task: Differentiate symmetric vs. asymmetric encryption\n"
        "Format: Output a 3-row comparison table followed by a 2-sentence summary"
    )
    assert needs_prediction(RTF, text) is False
    out = complete_from_prediction(RTF, text)
    assert "Role:" not in out
    assert "You are an enterprise information security architect." in out


def test_rtf_merge_fills_gaps_and_keeps_task():
    original = "What is the difference between synchronous and asynchronous encryption?"
    predicted = (
        "Role: enterprise information security architect\n"
        "Task: ignore this\n"
        "Format: a 3-row comparison table followed by a 2-sentence summary\n"
    )
    merged = merge_predicted_slots(RTF, original, predicted)
    assert merged["task"] == original
    assert "role" not in merged
    assert "3-row" not in merged.get("format", "").lower()
    assert "comparison" in merged["format"].lower()
    paragraph = complete_from_prediction(RTF, original, predicted)
    assert original.rstrip("?") in paragraph or "synchronous" in paragraph
    assert "You are" not in paragraph
    assert "3-row comparison table" not in paragraph
    assert "\n" not in paragraph


def test_rtf_why_question_omits_role_and_table():
    original = 'Why not remove "Role" specific from the prompt?'
    predicted = (
        "Role: subject-matter expert\n"
        "Task: ignore this\n"
        "Format: a 3-row comparison table followed by a 2-sentence summary\n"
    )
    paragraph = complete_from_prediction(RTF, original, predicted)
    assert "You are" not in paragraph
    assert "comparison table" not in paragraph.lower()
    assert "Role" in paragraph or "role" in paragraph.lower()


def test_tag_defaults_when_prediction_empty():
    original = "Walk me through how to set up an automated backup using cron and rsync."
    paragraph = complete_from_prediction(TAG, original, "")
    assert "Provide numbered setup steps" in paragraph
    assert "cron" in paragraph.lower() or "rsync" in paragraph.lower() or "Walk me" in paragraph


def test_create_and_grade_keep_user_task():
    create_in = "Draft three short headlines for an open-source analytics dashboard."
    merged = merge_predicted_slots(CREATE, create_in, "")
    assert merged["request"] == create_in
    grade_in = "I need a 4-week study plan to prepare for a cloud certification."
    merged_g = merge_predicted_slots(GRADE, grade_in, "")
    assert merged_g["goal"] == grade_in


def test_coast_trac_seed_role_from_act_as():
    text = (
        "Act as an interviewer conducting a technical screen for a systems "
        "architect role, and ask me three initial questions."
    )
    merged = merge_predicted_slots(COAST, text, "")
    assert "interviewer" in merged["actor"].lower()
    paragraph = complete_from_prediction(COAST, text, "")
    assert paragraph.startswith("You are an interviewer")
    trac = "Condense this policy briefing into a five-bullet executive summary."
    merged_t = merge_predicted_slots(TRAC, trac, "")
    assert merged_t["task"] == trac


BLENDER = (
    "I wanna write an email about my ninja blender, it stop working and I m pissed of."
)


def test_create_rejects_customer_support_for_complaint_email():
    predicted = (
        "Character: customer support representative\n"
        "Request: ignore this\n"
        "Examples: apologize, offer a solution, provide a timeframe\n"
        "Adjustments: Use clear, concise language\n"
        "Type: a concise list\n"
        "Extras: Include a link to the product page\n"
    )
    merged = merge_predicted_slots(CREATE, BLENDER, predicted)
    assert "character" not in merged
    assert merged["request"] == BLENDER
    assert "addressee" in merged["adjustments"].lower()
    assert merged["type"] == "a complete email"
    assert "emotion" in merged["extras"].lower()
    paragraph = complete_from_prediction(CREATE, BLENDER, predicted)
    assert not paragraph.startswith("You are")
    assert "customer support" not in paragraph.lower()
    assert "pissed" in paragraph.lower() or "blender" in paragraph.lower()


def test_create_keeps_anger_in_boss_email():
    original = "Can you help me draft an angry email to my boss?"
    predicted = (
        "Character: specialist copywriter\n"
        "Request: ignore this\n"
        "Examples: clear and professional\n"
        "Adjustments: No hype or clichés\n"
        "Type: a concise list\n"
        "Extras: Keep it brief\n"
    )
    merged = merge_predicted_slots(CREATE, original, predicted)
    assert "character" not in merged
    paragraph = complete_from_prediction(CREATE, original, predicted)
    assert "You are" not in paragraph
    assert "angry" in paragraph.lower()
    assert "clear and professional" not in paragraph.lower()


def test_create_mixi_keeps_product_and_disappointed():
    original = (
        "Write an email about my mixi not working. Express how disappointed I am."
    )
    predicted = (
        "Character: customer support representative\n"
        "Examples: frustrated, direct, and firm\n"
        "Adjustments: keep the user's stance as the customer; "
        "do not switch into customer support\n"
        "Type: a concise list\n"
        "Extras: preserve the frustration\n"
    )
    merged = merge_predicted_slots(CREATE, original, predicted)
    assert "character" not in merged
    paragraph = complete_from_prediction(CREATE, original, predicted)
    assert "You are" not in paragraph
    assert "mixi" in paragraph.lower()
    assert "disappointed" in paragraph.lower()
    assert "frustrated" not in paragraph.lower()
    assert "customer support" not in paragraph.lower()
    assert "[BANKING_DATA]" not in paragraph


def test_tag_generic_goal_for_non_technical_how_to():
    original = "Tell me step by step how to bake sourdough bread at home."
    paragraph = complete_from_prediction(TAG, original, "")
    assert "Provide numbered setup steps" in paragraph
    assert "command to verify" not in paragraph
    assert "concrete and easy to follow" in paragraph


def test_tag_keeps_verify_command_for_technical_how_to():
    original = "Walk me through how to set up an automated backup using cron and rsync."
    paragraph = complete_from_prediction(TAG, original, "")
    assert "command to verify success" in paragraph
