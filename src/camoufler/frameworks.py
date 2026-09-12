"""Prompt-expansion frameworks: RTF, TAG, CREATE, RACE, TRAC, COAST, GRADE."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from camoufler import race
from camoufler.slots import (
    ACTION_RE,
    EXPECT_RE,
    ROLE_EXTRACT_RE,
    ROLE_RE,
    USER_VOICE_ADJUSTMENTS,
    USER_VOICE_EMAIL_TYPE,
    USER_VOICE_EXTRAS,
    as_sentence,
    clean_predicted_value,
    echoes_prompt,
    extract_labeled_sections,
    format_role,
    is_user_voice_writing,
    matches_fewshot_ban,
    request_already_has_affect,
    role_inverts_speaker,
    should_omit_role,
    stated_tone_clause,
    uncap,
)

HasCheck = Callable[[str], bool]


@dataclass(frozen=True)
class Framework:
    """One expansion framework: slots, defaults, system prompt, assemble."""

    key: str
    acronym: str
    prompt_type: str
    slots: tuple[str, ...]
    labels: dict[str, str]
    defaults: dict[str, str]
    system_prompt: str
    task_slot: str
    assemble: Callable[[dict[str, str]], str]
    aliases: dict[str, str] = field(default_factory=dict)
    echo_slots: frozenset[str] = field(default_factory=frozenset)
    role_slot: str | None = None
    role_slots: frozenset[str] = field(default_factory=frozenset)
    verb_slots: frozenset[str] = field(default_factory=frozenset)
    has_checks: Mapping[str, HasCheck] = field(default_factory=dict)
    race_backend: bool = False
    fewshot_ban: frozenset[str] = field(default_factory=frozenset)


def _has_role(text: str) -> bool:
    return bool(ROLE_RE.search(text))


def _has_action(text: str) -> bool:
    return bool(ACTION_RE.search(text))


def _has_format(text: str) -> bool:
    return bool(EXPECT_RE.search(text)) or bool(
        re.search(r"(?i)\b(?:table|list|bullets?|json|markdown|summary)\b", text)
    )


def _has_question_task(text: str) -> bool:
    return _has_action(text) or bool(
        re.search(
            r"(?i)\b(?:what is|what's|what are|difference between|define|how many|"
            r"how much|who is|when was|versus|\bvs\.?\b)\b",
            text,
        )
    ) or text.strip().endswith("?")


def _has_how_to(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:how do i|how to|how can i|walk me through|step by step|"
            r"set\s*up|configure|install|tutorial)\b",
            text,
        )
    ) or _has_action(text)


def _has_steps(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:numbered|steps?|including|prerequisites|runbook)\b",
            text,
        )
    )


def _has_goal(text: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:ensure|verify|so that|confirm|success)\b", text)
    )


def _has_tone(text: str) -> bool:
    return bool(re.search(r"(?i)\b(?:tone|like|e\.g\.|crisp|professional)\b", text))


def _has_guardrail(text: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:avoid|don't|do not|no |without|guardrail|clich)\b", text)
    )


def _has_output_type(text: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:list|bullets?|headline|paragraph|table)\b", text)
    )


def _has_length(text: str) -> bool:
    return bool(re.search(r"(?i)\b(?:under \d+|words?|each |characters?)\b", text))


def _has_audience(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:audience|reader|exec|c-suite|stakeholder|for (?:c-suite|execs))\b",
            text,
        )
    )


def _has_constraints(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:maximum|don't|do not|zero|strict|bullet|no passive)\b",
            text,
        )
    )


def _has_setting(text: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:screening|interview|round|setting|simulation)\b", text)
    )


def _has_objective(text: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:assess|evaluate|objective|depth|resiliency)\b", text)
    ) or _has_action(text)


def _has_pacing(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:one question|wait|turn-by-turn|ask me|next question)\b",
            text,
        )
    )


def _has_tone_word(text: str) -> bool:
    return bool(re.search(r"(?i)\b(?:professional|rigorous|probing|tone)\b", text))


def _has_goal_plan(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:prepare|exam|cert|study plan|roadmap|goal)\b",
            text,
        )
    ) or _has_action(text)


def _has_assumptions(text: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:week|hours?|budget|timeline|intermediate)\b", text)
    )


def _has_deliverables(text: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:weekly|breakdown|labs?|plan|schedule)\b", text)
    )


def _has_eval(text: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:score|kpi|percent|%|benchmark|practice exam)\b", text)
    )


_TECHNICAL_RE = re.compile(
    r"(?i)\b(?:cron|rsync|bash|ssh|pytest|unit test|modulenotfounderror|"
    r"python|install|configure|pipeline|api|function|code)\b"
)
_TAG_TECH_GOAL = "each step includes a command to verify success"
_TAG_GENERIC_GOAL = "each step is concrete and easy to follow"
_CS_LEAK_RE = re.compile(
    r"(?i)\b(?:apologize|timeframe|offer a solution|product page|"
    r"customer support|no hype|clich)\b"
)
_PROFESSIONAL_TONE_RE = re.compile(r"(?i)\b(?:professional|clear and professional)\b")
_COMPARATIVE_RE = re.compile(
    r"(?i)\b(?:difference between|diff(?:erence)? between|versus|\bvs\.?\b|"
    r"compare|comparison)\b"
)


def _looks_technical(text: str) -> bool:
    return bool(_TECHNICAL_RE.search(text))


def _is_comparative(text: str) -> bool:
    return bool(_COMPARATIVE_RE.search(text))


def _drop_invented_role(fw: Framework, original: str, seeded: dict[str, str], answers: dict[str, str]) -> None:
    """Remove Role/Character/Actor unless the user assigned it."""
    if not fw.role_slot:
        return
    if should_omit_role(original, seeded, fw.role_slot):
        answers.pop(fw.role_slot, None)
        return
    role = answers.get(fw.role_slot, "")
    if role and role_inverts_speaker(role, original):
        answers.pop(fw.role_slot, None)


def _finalize_answers(
    fw: Framework, original: str, answers: dict[str, str], seeded: dict[str, str]
) -> dict[str, str]:
    """Omit invented roles, keep stated tone, and domain-limit TAG defaults."""
    _drop_invented_role(fw, original, seeded, answers)

    if fw.key == "create" and is_user_voice_writing(original):
        tone = stated_tone_clause(original)
        examples = answers.get("examples", "")
        leaked_examples = bool(
            examples
            and (
                _PROFESSIONAL_TONE_RE.search(examples)
                or _CS_LEAK_RE.search(examples)
                or matches_fewshot_ban(examples, fw.fewshot_ban)
            )
        )
        frustrated_upgrade = bool(
            examples
            and re.search(r"(?i)\bfrustrated\b", examples)
            and re.search(r"(?i)\b(?:disappointed|dissapointed|upset)\b", original)
        )
        if request_already_has_affect(original):
            if leaked_examples or frustrated_upgrade:
                answers.pop("examples", None)
        elif tone:
            answers["examples"] = tone
        elif leaked_examples:
            answers["examples"] = "Use a direct, firm tone"
        adjustments = answers.get("adjustments", "")
        if not _has_guardrail(original) or _CS_LEAK_RE.search(adjustments):
            answers["adjustments"] = USER_VOICE_ADJUSTMENTS
        if "email" in original.lower() and not _has_output_type(original):
            answers["type"] = USER_VOICE_EMAIL_TYPE
        if not _has_length(original):
            answers["extras"] = USER_VOICE_EXTRAS

    if fw.key == "tag":
        goal = answers.get("goal", "")
        if _looks_technical(original):
            if goal == _TAG_GENERIC_GOAL:
                answers["goal"] = _TAG_TECH_GOAL
        elif re.search(r"(?i)command to verify", goal):
            answers["goal"] = _TAG_GENERIC_GOAL
    return answers


def seed_answers(fw: Framework, text: str) -> dict[str, str]:
    """Fill slots already present so the model only supplies gaps."""
    if fw.race_backend:
        return race.seed_answers(text)
    answers = extract_labeled_sections(text, fw.slots, fw.aliases)
    if fw.task_slot not in answers:
        check = fw.has_checks.get(fw.task_slot, _has_action)
        if check(text):
            answers[fw.task_slot] = text.strip()
    if fw.role_slot and fw.role_slot not in answers:
        match = ROLE_EXTRACT_RE.search(text)
        if match:
            answers[fw.role_slot] = match.group(1).strip()
    return answers


def missing_slots(fw: Framework, text: str) -> list[str]:
    """Return missing slots in framework order."""
    if fw.race_backend:
        return race.missing_slots(text)
    labeled = extract_labeled_sections(text, fw.slots, fw.aliases)
    missing: list[str] = []
    for slot in fw.slots:
        if slot in labeled:
            continue
        check = fw.has_checks.get(slot)
        if check is not None and check(text):
            continue
        missing.append(slot)
    return missing


def needs_prediction(fw: Framework, text: str) -> bool:
    """True when a non-optional framework slot still needs a value."""
    if fw.race_backend:
        return race.needs_prediction(text)
    answers = seed_answers(fw, text)
    for slot in fw.slots:
        if slot in answers:
            continue
        if fw.role_slot and slot == fw.role_slot:
            continue
        if (
            fw.key == "rtf"
            and slot == "format"
            and not _is_comparative(text)
            and not _has_format(text)
        ):
            continue
        return True
    return False


def merge_predicted_slots(fw: Framework, original: str, predicted: str) -> dict[str, str]:
    """Keep user-supplied slots; fill the rest from the model, then defaults."""
    if fw.race_backend:
        return race.merge_predicted_slots(original, predicted)
    answers = seed_answers(fw, original)
    seeded = dict(answers)
    predicted_slots = extract_labeled_sections(predicted, fw.slots, fw.aliases)
    for slot in fw.slots:
        if slot in answers:
            continue
        raw = predicted_slots.get(slot, "")
        cleaned = clean_predicted_value(
            slot,
            raw,
            slot_names=fw.slots,
            aliases=fw.aliases,
            role_slots=fw.role_slots,
            verb_slots=fw.verb_slots,
        )
        if not cleaned:
            continue
        if slot in fw.echo_slots and echoes_prompt(cleaned, original):
            continue
        if matches_fewshot_ban(cleaned, fw.fewshot_ban):
            continue
        answers[slot] = cleaned
    if fw.task_slot not in answers:
        answers[fw.task_slot] = original.strip()
    for slot, fallback in fw.defaults.items():
        if fw.role_slot and slot == fw.role_slot:
            continue
        if fw.key == "rtf" and slot == "format":
            if _is_comparative(original) or _has_format(original):
                answers.setdefault(slot, fallback)
            continue
        if (
            fw.key == "create"
            and is_user_voice_writing(original)
            and slot in {"examples", "adjustments", "type", "extras"}
        ):
            continue
        answers.setdefault(slot, fallback)
    return _finalize_answers(fw, original, answers, seeded)


def complete_from_prediction(
    fw: Framework, original: str, predicted: str | None = None
) -> str:
    """Assemble a standard-English paragraph from the prompt plus optional model fill."""
    if fw.race_backend:
        return race.complete_from_prediction(original, predicted)
    if not needs_prediction(fw, original):
        return fw.assemble(seed_answers(fw, original))
    return fw.assemble(merge_predicted_slots(fw, original, predicted or ""))


def _ensure_lead(value: str, prefixes: tuple[str, ...], template: str) -> str:
    cleaned = value.strip().rstrip(".")
    if not cleaned:
        return ""
    if re.match(rf"(?i)^(?:{'|'.join(prefixes)})\b", cleaned):
        return as_sentence(cleaned)
    return as_sentence(template.format(uncap(cleaned)))


def _punctuate_join(*parts: str) -> str:
    """Join clauses, ensuring each ends with terminal punctuation."""
    sentences: list[str] = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        sentences.append(cleaned)
    return " ".join(sentences)


def assemble_rtf(answers: dict[str, str]) -> str:
    """Role, Task, Format → one paragraph. Role/format omitted when empty."""
    role = format_role(answers["role"]) if answers.get("role") else ""
    fmt = answers.get("format") or ""
    return _punctuate_join(
        role,
        as_sentence(answers.get("task") or ""),
        _ensure_lead(
            fmt,
            ("output", "provide", "use", "return", "give", "write", "format"),
            "Output {}",
        ),
    )


def assemble_tag(answers: dict[str, str]) -> str:
    """Task, Action, Goal → one paragraph."""
    return _punctuate_join(
        as_sentence(answers["task"]),
        as_sentence(answers["action"]),
        _ensure_lead(
            answers["goal"],
            ("ensure", "include", "verify", "confirm", "so that"),
            "Ensure {}",
        ),
    )


def assemble_create(answers: dict[str, str]) -> str:
    """CREATE slots → one paragraph. Character omitted when empty."""
    role = (
        format_role(answers["character"], default="specialist copywriter")
        if answers.get("character")
        else ""
    )
    return _punctuate_join(
        role,
        as_sentence(answers.get("request") or ""),
        _ensure_lead(
            answers.get("examples") or "",
            ("tone", "use", "crisp", "professional"),
            "Use a {} tone",
        ),
        as_sentence(answers.get("adjustments") or ""),
        _ensure_lead(
            answers.get("type") or "",
            ("output", "provide", "bulleted", "list"),
            "Output {}",
        ),
        as_sentence(answers.get("extras") or ""),
    )


def assemble_trac(answers: dict[str, str]) -> str:
    """Task, Role, Audience, Constraints → one paragraph."""
    task_core = (answers.get("task") or "").strip().rstrip(" .!?")
    audience = (answers.get("audience") or "").strip().rstrip(".")
    if audience and not re.match(r"(?i)^(for|aimed at)\b", audience):
        audience = f"for {audience}"
    role = format_role(answers["role"], default="editor") if answers.get("role") else ""
    task_line = f"{task_core} {audience}".strip() if audience else task_core
    return _punctuate_join(
        role,
        as_sentence(task_line),
        as_sentence(answers.get("constraints") or ""),
    )


def assemble_coast(answers: dict[str, str]) -> str:
    """Context, Objective, Actor, Scenario, Tone → one paragraph."""
    role = (
        format_role(answers["actor"], default="specialist")
        if answers.get("actor")
        else ""
    )
    return _punctuate_join(
        role,
        as_sentence(answers.get("context") or ""),
        as_sentence(answers.get("objective") or ""),
        as_sentence(answers.get("scenario") or ""),
        _ensure_lead(
            answers.get("tone") or "",
            ("tone", "keep", "remain"),
            "Keep the tone {}",
        ),
    )


def assemble_grade(answers: dict[str, str]) -> str:
    """Goal, Role, Assumptions, Deliverables, Evaluation → one paragraph."""
    role = (
        format_role(answers["role"], default="strategist")
        if answers.get("role")
        else ""
    )
    return _punctuate_join(
        role,
        as_sentence(answers.get("goal") or ""),
        _ensure_lead(
            answers.get("assumptions") or "",
            ("assumptions", "given", "with"),
            "Assumptions: {}",
        ),
        as_sentence(answers.get("deliverables") or ""),
        _ensure_lead(
            answers.get("evaluation") or "",
            ("benchmark", "success", "measure", "success:"),
            "Success: {}",
        ),
    )


def assemble_race(answers: dict[str, str]) -> str:
    """Delegate to the RACE assembler."""
    return race.assemble_race(answers)


RTF_SYSTEM = (
    "You infer missing RTF slots for a prompt. You never fulfill the request.\n\n"
    "Infer Role, Task, and Format from the input. Output only three labeled lines.\n"
    "Do not answer the question. Do not write the comparison, table, or definition.\n\n"
    "Output exactly:\n"
    "Role: ...\n"
    "Task: ...\n"
    "Format: ...\n\n"
    "Example:\n"
    "Input: What is the difference between synchronous and asynchronous encryption?\n"
    "Role: enterprise information security architect\n"
    "Task: differentiate symmetric vs. asymmetric encryption across key exchange, "
    "computational overhead, and common protocols\n"
    "Format: a direct 3-row comparison table followed by a 2-sentence summary"
)

TAG_SYSTEM = (
    "You infer missing TAG slots for a prompt. You never fulfill the request.\n\n"
    "Infer Task, Action, and Goal. Output only three labeled lines.\n"
    "Do not write the tutorial or run the commands.\n\n"
    "Output exactly:\n"
    "Task: ...\n"
    "Action: ...\n"
    "Goal: ...\n\n"
    "Example:\n"
    "Input: Walk me through how to set up an automated backup pipeline using cron and rsync.\n"
    "Task: configure an automated incremental server backup using rsync and cron\n"
    "Action: provide numbered setup steps including SSH key authentication, log rotation, and syntax\n"
    "Goal: each step includes a bash command to verify successful execution"
)

CREATE_SYSTEM = (
    "You infer missing CREATE slots for a prompt. You never fulfill the request.\n\n"
    "Infer Character, Request, Examples, Adjustments, Type, and Extras. "
    "Output only six labeled lines. Do not write the headlines or copy.\n\n"
    "Leave Character empty unless the user said act as or you are. "
    "If the user is writing in their own voice (complaint, email, 'I want to write'), "
    "do not invent a Character and never use the addressee (customer support, the boss).\n"
    "Keep stated emotion in Examples. Do not upgrade disappointed to frustrated.\n\n"
    "Output exactly:\n"
    "Character: ...\n"
    "Request: ...\n"
    "Examples: ...\n"
    "Adjustments: ...\n"
    "Type: ...\n"
    "Extras: ...\n\n"
    "Example:\n"
    "Input: Draft three short headlines for an open-source analytics dashboard launch.\n"
    "Character: direct-response B2B copywriter\n"
    "Request: draft 3 product launch headlines for an internal observability dashboard\n"
    "Examples: crisp, technical, and benefit-driven\n"
    "Adjustments: no marketing jargon or hype words\n"
    "Type: bulleted list with sub-bullets explaining value proposition\n"
    "Extras: keep each headline under 10 words\n\n"
    "Example:\n"
    "Input: I wanna write an email about my ninja blender, it stopped working "
    "and I'm pissed off.\n"
    "Character:\n"
    "Request: write an email about my Ninja blender that stopped working\n"
    "Examples: frustrated, direct, and firm\n"
    "Adjustments: keep the user's stance; do not switch into the addressee's role\n"
    "Type: a complete email\n"
    "Extras: preserve the user's emotion and stance"
)

TRAC_SYSTEM = (
    "You infer missing TRAC slots for a prompt. You never fulfill the request.\n\n"
    "Infer Task, Role, Audience, and Constraints. Output only four labeled lines.\n"
    "Do not write the summary or rewrite the source beyond slot inference.\n\n"
    "Output exactly:\n"
    "Task: ...\n"
    "Role: ...\n"
    "Audience: ...\n"
    "Constraints: ...\n\n"
    "Example:\n"
    "Input: Condense this policy briefing into a five-bullet executive summary on compliance risks.\n"
    "Task: condense the attached briefing into an executive summary\n"
    "Role: strategic management consultant\n"
    "Audience: C-suite executives with limited technical time\n"
    "Constraints: maximum 5 bullet points, focus on compliance risk, do not add facts"
)

COAST_SYSTEM = (
    "You infer missing COAST slots for a prompt. You never fulfill the request.\n\n"
    "Infer Context, Objective, Actor, Scenario, and Tone. Output only five labeled lines.\n"
    "Do not start the interview or ask the screening questions.\n\n"
    "Output exactly:\n"
    "Context: ...\n"
    "Objective: ...\n"
    "Actor: ...\n"
    "Scenario: ...\n"
    "Tone: ...\n\n"
    "Example:\n"
    "Input: Act as an interviewer conducting a technical screen for a systems architect role, "
    "and ask me three initial questions.\n"
    "Context: technical screening round for a Senior Cloud Infrastructure Lead\n"
    "Objective: assess candidate depth in distributed system resiliency\n"
    "Actor: principal infrastructure architect\n"
    "Scenario: ask one scenario-based question at a time and wait for the candidate response\n"
    "Tone: professional, rigorous, and probing"
)

GRADE_SYSTEM = (
    "You infer missing GRADE slots for a prompt. You never fulfill the request.\n\n"
    "Infer Goal, Role, Assumptions, Deliverables, and Evaluation. "
    "Output only five labeled lines. Do not write the study plan.\n\n"
    "Output exactly:\n"
    "Goal: ...\n"
    "Role: ...\n"
    "Assumptions: ...\n"
    "Deliverables: ...\n"
    "Evaluation: ...\n\n"
    "Example:\n"
    "Input: I need a 4-week study plan to prepare for a cloud certification, studying roughly 6 hours per week.\n"
    "Goal: prepare for the AWS Certified Solutions Architect - Associate exam\n"
    "Role: cloud certification mentor\n"
    "Assumptions: 4-week timeline, 6 hours/week study budget, intermediate networking foundation\n"
    "Deliverables: weekly study breakdown, recommended hands-on labs, and practice exams\n"
    "Evaluation: scoring 85%+ on official timed practice assessments"
)


RTF = Framework(
    key="rtf",
    acronym="RTF",
    prompt_type="factual",
    slots=("role", "task", "format"),
    labels={"role": "Role", "task": "Task", "format": "Format"},
    defaults={
        "format": "a concise comparison",
    },
    system_prompt=RTF_SYSTEM,
    task_slot="task",
    assemble=assemble_rtf,
    echo_slots=frozenset({"format"}),
    role_slot="role",
    role_slots=frozenset({"role"}),
    verb_slots=frozenset({"task"}),
    has_checks={
        "role": _has_role,
        "task": _has_question_task,
        "format": _has_format,
    },
    fewshot_ban=frozenset(
        {
            "enterprise information security architect",
            "3 row comparison table",
            "2 sentence summary",
        }
    ),
)

TAG = Framework(
    key="tag",
    acronym="TAG",
    prompt_type="instructional",
    slots=("task", "action", "goal"),
    labels={"task": "Task", "action": "Action", "goal": "Goal"},
    defaults={
        "action": "Provide numbered setup steps",
        "goal": "each step is concrete and easy to follow",
    },
    system_prompt=TAG_SYSTEM,
    task_slot="task",
    assemble=assemble_tag,
    echo_slots=frozenset({"action", "goal"}),
    verb_slots=frozenset({"task", "action"}),
    has_checks={
        "task": _has_how_to,
        "action": _has_steps,
        "goal": _has_goal,
    },
)

CREATE = Framework(
    key="create",
    acronym="CREATE",
    prompt_type="creative",
    slots=("character", "request", "examples", "adjustments", "type", "extras"),
    labels={
        "character": "Character",
        "request": "Request",
        "examples": "Examples",
        "adjustments": "Adjustments",
        "type": "Type",
        "extras": "Extras",
    },
    defaults={
        "character": "specialist copywriter",
        "examples": "clear and professional",
        "adjustments": "No hype or clichés",
        "type": "a concise list",
        "extras": "Keep it brief",
    },
    system_prompt=CREATE_SYSTEM,
    task_slot="request",
    assemble=assemble_create,
    echo_slots=frozenset({"examples", "adjustments", "type", "extras"}),
    role_slot="character",
    role_slots=frozenset({"character"}),
    verb_slots=frozenset({"request"}),
    has_checks={
        "character": _has_role,
        "request": _has_action,
        "examples": _has_tone,
        "adjustments": _has_guardrail,
        "type": _has_output_type,
        "extras": _has_length,
    },
    fewshot_ban=frozenset(
        {
            "writing assistant helping the user draft",
            "direct response b2b copywriter",
            "do not switch into customer support",
            "preserve the frustration",
            "keep the user s stance as the customer",
        }
    ),
)

RACE = Framework(
    key="race",
    acronym="RACE",
    prompt_type="analytical",
    slots=race.SLOTS,
    labels=dict(race.SLOT_LABELS),
    defaults={
        "role": "a specialist in the topic",
        "context": "a general audience",
        "expectations": "clear and concise",
    },
    system_prompt=race.DEFAULT_RACE_SYSTEM,
    task_slot="action",
    assemble=assemble_race,
    aliases={"expectation": "expectations"},
    echo_slots=frozenset({"context", "expectations"}),
    role_slot="role",
    role_slots=frozenset({"role"}),
    verb_slots=frozenset({"action"}),
    race_backend=True,
)

TRAC = Framework(
    key="trac",
    acronym="TRAC",
    prompt_type="transformation",
    slots=("task", "role", "audience", "constraints"),
    labels={
        "task": "Task",
        "role": "Role",
        "audience": "Audience",
        "constraints": "Constraints",
    },
    defaults={
        "role": "editor",
        "audience": "a general audience",
        "constraints": "Keep fidelity to the source. Do not add facts.",
    },
    system_prompt=TRAC_SYSTEM,
    task_slot="task",
    assemble=assemble_trac,
    echo_slots=frozenset({"audience", "constraints"}),
    role_slot="role",
    role_slots=frozenset({"role"}),
    verb_slots=frozenset({"task"}),
    has_checks={
        "task": _has_action,
        "role": _has_role,
        "audience": _has_audience,
        "constraints": _has_constraints,
    },
    fewshot_ban=frozenset({"strategic management consultant"}),
)

COAST = Framework(
    key="coast",
    acronym="COAST",
    prompt_type="roleplay",
    slots=("context", "objective", "actor", "scenario", "tone"),
    labels={
        "context": "Context",
        "objective": "Objective",
        "actor": "Actor",
        "scenario": "Scenario",
        "tone": "Tone",
    },
    defaults={
        "context": "a professional simulation",
        "actor": "specialist in the topic",
        "scenario": "Ask one question at a time and wait for the response",
        "tone": "professional and rigorous",
    },
    system_prompt=COAST_SYSTEM,
    task_slot="objective",
    assemble=assemble_coast,
    echo_slots=frozenset({"context", "scenario", "tone"}),
    role_slot="actor",
    role_slots=frozenset({"actor"}),
    verb_slots=frozenset({"objective"}),
    has_checks={
        "context": _has_setting,
        "objective": _has_objective,
        "actor": _has_role,
        "scenario": _has_pacing,
        "tone": _has_tone_word,
    },
    fewshot_ban=frozenset(
        {
            "principal infrastructure architect",
            "senior cloud infrastructure lead",
        }
    ),
)

GRADE = Framework(
    key="grade",
    acronym="GRADE",
    prompt_type="strategic",
    slots=("goal", "role", "assumptions", "deliverables", "evaluation"),
    labels={
        "goal": "Goal",
        "role": "Role",
        "assumptions": "Assumptions",
        "deliverables": "Deliverables",
        "evaluation": "Evaluation",
    },
    defaults={
        "role": "strategist",
        "assumptions": "a limited time budget",
        "deliverables": "a weekly milestone plan",
        "evaluation": "measurable checkpoints",
    },
    system_prompt=GRADE_SYSTEM,
    task_slot="goal",
    assemble=assemble_grade,
    echo_slots=frozenset({"assumptions", "deliverables", "evaluation"}),
    role_slot="role",
    role_slots=frozenset({"role"}),
    verb_slots=frozenset({"goal"}),
    has_checks={
        "goal": _has_goal_plan,
        "role": _has_role,
        "assumptions": _has_assumptions,
        "deliverables": _has_deliverables,
        "evaluation": _has_eval,
    },
    fewshot_ban=frozenset(
        {
            "cloud certification mentor",
            "aws certified solutions architect",
        }
    ),
)

FRAMEWORKS: dict[str, Framework] = {
    "rtf": RTF,
    "tag": TAG,
    "create": CREATE,
    "race": RACE,
    "trac": TRAC,
    "coast": COAST,
    "grade": GRADE,
}

TYPE_TO_FRAMEWORK = {
    "factual": RTF,
    "instructional": TAG,
    "creative": CREATE,
    "analytical": RACE,
    "transformation": TRAC,
    "roleplay": COAST,
    "strategic": GRADE,
}
