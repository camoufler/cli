"""Shared labeled-slot parsing, cleaning, and sentence helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

JUNK_RE = re.compile(r"(?i)^(none|null|n/?a|nil|undefined|nan|-|none\.|null\.)$")
ROLE_RE = re.compile(
    r"(?i)(?:\byou are\b|\bact as\b|\bacting as\b|\brole\s*:|"
    r"\bas an?\s+[a-z][a-z\-']{2,})"
)
ACTION_RE = re.compile(
    r"(?i)\b(?:"
    r"analy[sz]e|synthesi[sz]e|compare|evaluate|draft|rewrite|"
    r"optimize|condense|debug|refactor|architect|review|"
    r"explain|teach|translate|summarize|write|create|generate|"
    r"design|produce|redact|normalize|standardize|outline|"
    r"implement|fix|plan|build|proof\s*read|proofread|edit|"
    r"correct|paraphrase|rephrase"
    r")\b"
)
SLANG_RE = re.compile(
    r"(?i)\b(?:lowkey|bussin|veg out|spilling the tea|finna|gonna|"
    r"wanna|gotta|sus|no cap|yo,)\b"
)
TASK_RE = re.compile(
    r"(?i)\b(?:proof\s*read|rewrite|write|draft|create|generate|"
    r"blog|email|analyze|summarize)\b"
)
CONTEXT_RE = re.compile(
    r"(?i)\b(?:"
    r"audience|users?|because|constraint|background|"
    r"i(?:'m| am) building|we need|target|"
    r"for (?:senior|junior|beginner|developers?|engineers?|"
    r"execs?|kids|non-technical)|"
    r"desktop|mobile|deadline|budget|context\s*:"
    r")\b"
)
EXPECT_RE = re.compile(
    r"(?i)\b(?:"
    r"json|markdown|bullet|tone|under \d+|sections?|"
    r"output as|format|keep it|include|don'?t use|"
    r"expectations?\s*:|headlines?|outline|word\s*counts?|"
    r"paragraphs?|word count"
    r")\b"
)
ROLE_EXTRACT_RE = re.compile(r"(?is)((?:you are|act as)\s+.+?)(?:\.|$)")
ROLE_STOPWORDS = frozenset(
    {
        "action",
        "context",
        "expectation",
        "expectations",
        "role",
        "write",
        "none",
        "null",
        "narrator",
    }
)


def tidy(value: str) -> str:
    """Collapse whitespace and normalize commas."""
    cleaned = " ".join(value.strip().split())
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    return cleaned.strip(" ,")


def uncap(value: str) -> str:
    """Lowercase the first character."""
    if not value:
        return value
    return value[0].lower() + value[1:]


def as_sentence(value: str) -> str:
    """Title-case the first letter and end with a period unless already punctuated."""
    cleaned = tidy(value).rstrip(" .")
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned[-1] in ".!?":
        return cleaned
    return f"{cleaned}."


def is_junk(value: str) -> bool:
    """True for empty or dummy model values like None/n/a."""
    return not value or bool(JUNK_RE.match(tidy(value)))


def is_utterance(text: str) -> bool:
    """True when the input is slang/content to rewrite, not a generation ask."""
    if TASK_RE.search(text):
        return False
    return bool(SLANG_RE.search(text)) or not bool(ACTION_RE.search(text))


def is_proofread(text: str) -> bool:
    """True when the user asked to proofread."""
    return bool(re.search(r"(?i)proof\s*read", text))


def echoes_prompt(value: str, original: str) -> bool:
    """True when the model just repeated the user's ask."""
    a = re.sub(r"\W+", " ", value.lower()).strip()
    b = re.sub(r"\W+", " ", original.lower()).strip()
    if not a or not b:
        return True
    if a == b or a in b or b in a:
        return True
    sa, sb = set(a.split()), set(b.split())
    if not sa:
        return True
    return len(sa & sb) / len(sa) >= 0.8


def _labeled_re(slot_names: Sequence[str], aliases: Mapping[str, str] | None) -> re.Pattern[str]:
    names = {n.lower() for n in slot_names}
    if aliases:
        names.update(k.lower() for k in aliases)
    pattern = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?im)^({pattern})\s*:\s*(.*)$")


def _canonical_slot(key: str, aliases: Mapping[str, str] | None) -> str:
    key = key.lower()
    if aliases and key in aliases:
        return aliases[key]
    if key.startswith("expectation"):
        return "expectations"
    return key


def extract_labeled_sections(
    text: str,
    slot_names: Sequence[str],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Parse `Label: value` lines for the given slot names."""
    labeled = _labeled_re(slot_names, aliases)
    found: dict[str, str] = {}
    current: str | None = None
    chunks: dict[str, list[str]] = {}
    for raw in text.splitlines():
        match = labeled.match(raw.strip())
        if match:
            current = _canonical_slot(match.group(1), aliases)
            chunks[current] = [match.group(2).strip()]
            continue
        if current and raw.strip():
            chunks[current].append(raw.strip())
    for key, parts in chunks.items():
        value = " ".join(p for p in parts if p).strip()
        if value:
            found[key] = value
    return found


def nested_label_re(slot_names: Sequence[str], aliases: Mapping[str, str] | None = None) -> re.Pattern[str]:
    """Match a leaked `Slot:` prefix at the start of a predicted value."""
    names = {n.lower() for n in slot_names}
    if aliases:
        names.update(k.lower() for k in aliases)
    pattern = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?i)^({pattern})\s*:\s*")


def clean_predicted_value(
    slot: str,
    value: str,
    *,
    slot_names: Sequence[str],
    aliases: Mapping[str, str] | None = None,
    role_slots: frozenset[str] | None = None,
    verb_slots: frozenset[str] | None = None,
) -> str:
    """Drop leaked slot labels and dummy values like None."""
    role_slots = role_slots or frozenset()
    verb_slots = verb_slots or frozenset()
    nested = nested_label_re(slot_names, aliases)
    cleaned = tidy(value)
    if is_junk(cleaned):
        return ""
    while True:
        match = nested.match(cleaned)
        if not match:
            break
        nested_slot = _canonical_slot(match.group(1), aliases)
        if nested_slot != slot:
            return ""
        cleaned = cleaned[match.end() :].strip()
    if is_junk(cleaned):
        return ""
    if slot in role_slots:
        if cleaned.lower() in ROLE_STOPWORDS:
            return ""
        if ACTION_RE.fullmatch(cleaned):
            return ""
    if slot in verb_slots and ACTION_RE.fullmatch(cleaned):
        return ""
    return cleaned


def with_article(value: str) -> str:
    """Prefix a/an unless the value already has an article."""
    cleaned = tidy(value).rstrip(".")
    if not re.match(r"(?i)^(a|an|the)\b", cleaned):
        article = "an" if re.match(r"(?i)^[aeiou]", cleaned) else "a"
        cleaned = f"{article} {cleaned}"
    return cleaned


def format_role(value: str, default: str = "copy editor") -> str:
    """`You are a {role}.`"""
    cleaned = tidy(value).rstrip(".")
    cleaned = re.sub(r"(?i)^(you are|act as)\s+", "", cleaned)
    if is_junk(cleaned) or cleaned.lower() in ROLE_STOPWORDS:
        cleaned = default
    cleaned = with_article(cleaned)
    return f"You are {cleaned}."


def format_action(value: str, default: str = "Rewrite the text into standard English") -> str:
    """Sentence-case an action/task clause."""
    cleaned = tidy(value).rstrip(".")
    if is_junk(cleaned):
        cleaned = default
    cleaned = re.sub(r"(?i)^write me\b", "write", cleaned)
    cleaned = re.sub(r"(?i)^proof\s+read\b", "Proofread", cleaned)
    return as_sentence(cleaned)


def format_context(value: str, default: str = "a general audience") -> str:
    """`for {audience}` unless already prepositioned."""
    cleaned = tidy(value).rstrip(".")
    if is_junk(cleaned):
        cleaned = default
    if re.match(r"(?i)^(for|aimed at|audience)\b", cleaned):
        return cleaned
    return f"for {uncap(cleaned)}"


def format_expectations(value: str, default: str = "clear and concise") -> str:
    """Sentence that states output constraints."""
    cleaned = tidy(value).rstrip(".")
    if is_junk(cleaned):
        cleaned = default
    if re.match(r"(?i)^(keep|deliver|output|give|use|include|the result)\b", cleaned):
        return as_sentence(cleaned)
    if re.search(r"(?i)\b(is to be|should be|must be|the text)\b", cleaned):
        return as_sentence(cleaned)
    return as_sentence(f"Keep it {uncap(cleaned)}")
