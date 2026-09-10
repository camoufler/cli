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
    as_sentence,
    clean_predicted_value,
    echoes_prompt,
    extract_labeled_sections,
    format_role,
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
    """True when any framework slot still needs a value."""
    if fw.race_backend:
        return race.needs_prediction(text)
    answers = seed_answers(fw, text)
    return any(slot not in answers for slot in fw.slots)


def merge_predicted_slots(fw: Framework, original: str, predicted: str) -> dict[str, str]:
    """Keep user-supplied slots; fill the rest from the model, then defaults."""
    if fw.race_backend:
        return race.merge_predicted_slots(original, predicted)
    answers = seed_answers(fw, original)
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
        answers[slot] = cleaned
    if fw.task_slot not in answers:
        answers[fw.task_slot] = original.strip()
    if fw.role_slot and fw.role_slot not in answers:
        answers[fw.role_slot] = fw.defaults.get(fw.role_slot, "a specialist in the topic")
    for slot, fallback in fw.defaults.items():
        answers.setdefault(slot, fallback)
    return answers


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
    """Role, Task, Format → one paragraph."""
    return _punctuate_join(
        format_role(answers["role"], default="subject-matter expert"),
        as_sentence(answers["task"]),
        _ensure_lead(
            answers["format"],
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
    """CREATE slots → one paragraph."""
    return _punctuate_join(
        format_role(answers["character"], default="specialist copywriter"),
        as_sentence(answers["request"]),
        _ensure_lead(
            answers["examples"],
            ("tone", "use", "crisp", "professional"),
            "Use a {} tone",
        ),
        as_sentence(answers["adjustments"]),
        _ensure_lead(
            answers["type"],
            ("output", "provide", "bulleted", "list"),
            "Output {}",
        ),
        as_sentence(answers["extras"]),
    )


def assemble_trac(answers: dict[str, str]) -> str:
    """Task, Role, Audience, Constraints → one paragraph."""
    task_core = answers["task"].strip().rstrip(" .!?")
    audience = answers["audience"].strip().rstrip(".")
    if not re.match(r"(?i)^(for|aimed at)\b", audience):
        audience = f"for {audience}"
    return _punctuate_join(
        format_role(answers["role"], default="editor"),
        as_sentence(f"{task_core} {audience}"),
        as_sentence(answers["constraints"]),
    )


def assemble_coast(answers: dict[str, str]) -> str:
    """Context, Objective, Actor, Scenario, Tone → one paragraph."""
    return _punctuate_join(
        format_role(answers["actor"], default="specialist"),
        as_sentence(answers["context"]),
        as_sentence(answers["objective"]),
        as_sentence(answers["scenario"]),
        _ensure_lead(
            answers["tone"],
            ("tone", "keep", "remain"),
            "Keep the tone {}",
        ),
    )


def assemble_grade(answers: dict[str, str]) -> str:
    """Goal, Role, Assumptions, Deliverables, Evaluation → one paragraph."""
    return _punctuate_join(
        format_role(answers["role"], default="strategist"),
        as_sentence(answers["goal"]),
        _ensure_lead(
            answers["assumptions"],
            ("assumptions", "given", "with"),
            "Assumptions: {}",
        ),
        as_sentence(answers["deliverables"]),
        _ensure_lead(
            answers["evaluation"],
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
    "Extras: keep each headline under 10 words"
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
        "role": "subject-matter expert",
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
)

TAG = Framework(
    key="tag",
    acronym="TAG",
    prompt_type="instructional",
    slots=("task", "action", "goal"),
    labels={"task": "Task", "action": "Action", "goal": "Goal"},
    defaults={
        "action": "Provide numbered setup steps",
        "goal": "each step includes a command to verify success",
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
