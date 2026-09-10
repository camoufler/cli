"""Detect missing RACE slots, merge model predictions, assemble a paragraph."""

from __future__ import annotations

from camoufler.slots import (
    ACTION_RE,
    CONTEXT_RE,
    EXPECT_RE,
    ROLE_EXTRACT_RE,
    ROLE_RE,
    clean_predicted_value as _clean_slot_value,
    echoes_prompt as _echoes_prompt,
    extract_labeled_sections as _extract_labeled,
    format_action as _format_action,
    format_context as _format_context,
    format_expectations as _format_expectations,
    format_role as _format_role,
    is_proofread as _is_proofread,
    is_utterance as _is_utterance,
)

SLOTS = ("role", "action", "context", "expectations")
SLOT_LABELS = {
    "role": "Role",
    "action": "Action",
    "context": "Context",
    "expectations": "Expectations",
}
_ALIASES = {"expectation": "expectations"}

DEFAULT_RACE_SYSTEM = (
    "Role: You infer missing RACE slots for a prompt. You never fulfill the request.\n\n"
    "Action: Infer Role, Action, Context, and Expectations from the input. "
    "Output only four labeled lines.\n\n"
    "Context: The input is an incomplete prompt. Infer a fitting specialist Role, "
    "keep the user's task as Action, infer audience as Context, and infer "
    "length, tone, and format as Expectations. Do not write the blog, email, "
    "or code they asked for.\n\n"
    "Expectation: Output exactly:\n"
    "Role: ...\n"
    "Action: ...\n"
    "Context: ...\n"
    "Expectations: ...\n\n"
    "Example:\n"
    "Input: Write me a blog post about AI\n"
    "Role: technical writer\n"
    "Action: write a blog post about AI\n"
    "Context: general readers new to AI\n"
    "Expectations: short, simple, and clear\n\n"
    "Example:\n"
    "Input: Proof read this slang paragraph\n"
    "Role: copy editor\n"
    "Action: proofread the quoted text into standard English\n"
    "Context: informal spoken English\n"
    "Expectations: keep the meaning and similar length"
)

DEFAULT_RACE_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "num_predict": 256,
}

_DEFAULTS = {
    "role": "a specialist in the topic",
    "context": "a general audience",
    "expectations": "clear and concise",
}


def extract_labeled_sections(text: str) -> dict[str, str]:
    """Parse Role:/Action:/Context:/Expectation(s): lines."""
    return _extract_labeled(text, SLOTS, _ALIASES)


def _has_role(text: str) -> bool:
    return bool(ROLE_RE.search(text))


def _has_action(text: str) -> bool:
    return bool(ACTION_RE.search(text))


def _has_context(text: str) -> bool:
    return bool(CONTEXT_RE.search(text))


def _has_expectations(text: str) -> bool:
    return bool(EXPECT_RE.search(text))


def missing_slots(text: str) -> list[str]:
    """Return missing RACE slots in Role, Action, Context, Expectations order."""
    labeled = extract_labeled_sections(text)
    checks = {
        "role": _has_role,
        "action": _has_action,
        "context": _has_context,
        "expectations": _has_expectations,
    }
    missing: list[str] = []
    for slot in SLOTS:
        if slot in labeled:
            continue
        if not checks[slot](text):
            missing.append(slot)
    return missing


def seed_answers(text: str) -> dict[str, str]:
    """Fill slots already present so the model only supplies gaps."""
    answers = extract_labeled_sections(text)
    if "action" not in answers and _has_action(text):
        answers["action"] = text.strip()
    if "role" not in answers:
        match = ROLE_EXTRACT_RE.search(text)
        if match:
            answers["role"] = match.group(1).strip()
    return answers


def needs_prediction(text: str) -> bool:
    """True when Role, Context, or Expectations (or Action) still need values."""
    answers = seed_answers(text)
    return any(slot not in answers for slot in SLOTS)


def _guess_role(original: str) -> str:
    lower = original.lower()
    if _is_utterance(original) or _is_proofread(original):
        return "copy editor"
    if "blog" in lower or "article" in lower or "post" in lower:
        return "technical writer"
    if "email" in lower:
        return "copywriter"
    if any(token in lower for token in ("code", "function", "api", "python", "debug")):
        return "software engineer"
    return _DEFAULTS["role"]


def _clean_predicted_value(slot: str, value: str) -> str:
    """Drop leaked slot labels and dummy values like None."""
    return _clean_slot_value(
        slot,
        value,
        slot_names=SLOTS,
        aliases=_ALIASES,
        role_slots=frozenset({"role"}),
        verb_slots=frozenset({"action"}),
    )


def _canonical_action(original: str) -> str:
    if _is_proofread(original):
        return "Proofread the quoted text into standard English"
    if _is_utterance(original):
        return "Rewrite the text into standard English"
    return original.strip()


def merge_predicted_slots(original: str, predicted: str) -> dict[str, str]:
    """Keep user-supplied slots; fill the rest from the model, then defaults."""
    answers = seed_answers(original)
    edit_job = _is_utterance(original) or _is_proofread(original)
    if edit_job:
        answers.pop("action", None)
        answers.pop("role", None)
    predicted_slots = extract_labeled_sections(predicted)
    for slot in SLOTS:
        if slot in answers:
            continue
        raw = predicted_slots.get(slot, "")
        cleaned = _clean_predicted_value(slot, raw)
        if not cleaned:
            continue
        if slot in {"context", "expectations"} and _echoes_prompt(cleaned, original):
            continue
        answers[slot] = cleaned
    if edit_job:
        answers["role"] = _guess_role(original)
        answers["action"] = _canonical_action(original)
        answers["context"] = "informal spoken English"
        answers["expectations"] = "keep the meaning and similar length"
        return answers
    if "action" not in answers:
        answers["action"] = _canonical_action(original)
    if "role" not in answers:
        answers["role"] = _guess_role(original)
    for slot, fallback in _DEFAULTS.items():
        answers.setdefault(slot, fallback)
    return answers


def complete_from_prediction(original: str, predicted: str | None = None) -> str:
    """Assemble a standard-English paragraph from the prompt plus optional model fill."""
    if not needs_prediction(original):
        return assemble_race(seed_answers(original))
    if not predicted:
        return assemble_race(merge_predicted_slots(original, ""))
    return assemble_race(merge_predicted_slots(original, predicted))


def assemble_race(answers: dict[str, str]) -> str:
    """Fold Role, Action, Context, and Expectations into one English paragraph."""
    role = _format_role(answers["role"]).rstrip(".")
    action = _format_action(answers["action"]).rstrip(".")
    context = _format_context(answers["context"])
    expectations = _format_expectations(answers["expectations"])
    return f"{role}. {action} {context}. {expectations}"
