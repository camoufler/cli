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
EXPLICIT_ROLE_RE = re.compile(r"(?i)\b(?:act as|you are|role-?play)\b")
USER_VOICE_WRITING_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:i wanna|i want to|i need to|i am going to|i['’]m gonna)\s+write\b|"
    r"\bwrite (?:an? |me )?(?:email|letter|complaint|message)\b|"
    r"\bdraft (?:an? )?(?:angry |frustrated |pissed[- ]off )?(?:email|letter|complaint)\b|"
    r"\b(?:angry|frustrated|pissed(?:\s+off)?)\s+email\b|"
    r"\b(?:email|letter|complaint)\b.+\b(?:my|i['’]m|i am|pissed|stopped working)\b|"
    r"\bmy \w+(?:\s+\w+){0,3}\s+(?:stopped working|broke|is broken)\b|"
    r"\bcomplain(?:t|ing)?\b"
    r")"
)
ADDRESSEE_ROLE_RE = re.compile(
    r"(?i)\b(?:"
    r"customer support|customer service|support representative|"
    r"service representative|support agent|technical support|"
    r"support specialist|call center|interviewer|"
    r"(?:the )?boss|vendor|manufacturer"
    r")\b"
)
AFFECT_PISSED_RE = re.compile(r"(?i)\b(?:pissed|furious|livid)\b")
AFFECT_ANGRY_RE = re.compile(r"(?i)\bangry\b")
AFFECT_FRUSTRATED_RE = re.compile(r"(?i)\bfrustrated\b")
AFFECT_DISAPPOINTED_RE = re.compile(r"(?i)\b(?:disappointed|dissapointed|upset)\b")
AFFECT_ANY_RE = re.compile(
    r"(?i)\b(?:express how|disappointed|dissapointed|upset|angry|"
    r"frustrated|pissed|furious|livid)\b"
)
USER_VOICE_ROLE = "writing assistant helping the user draft in their own voice"
USER_VOICE_ADJUSTMENTS = (
    "Keep the user's stance; do not switch into the addressee's role"
)
USER_VOICE_EXTRAS = "Preserve the user's emotion and stance"
USER_VOICE_EMAIL_TYPE = "a complete email"


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


def is_explicit_role_request(text: str) -> bool:
    """True when the user assigned the model a role (act as / you are)."""
    return bool(EXPLICIT_ROLE_RE.search(text))


def is_user_voice_writing(text: str) -> bool:
    """True when the user wants output in their own voice (complaint, email)."""
    return bool(USER_VOICE_WRITING_RE.search(text))


def role_inverts_speaker(role: str, original: str) -> bool:
    """True when a predicted role is the addressee of a first-person ask."""
    if is_explicit_role_request(original) or not is_user_voice_writing(original):
        return False
    return bool(ADDRESSEE_ROLE_RE.search(role))


def helper_writing_role(original: str) -> str | None:
    """Marker for first-person writing with no assigned role (do not inject it)."""
    if is_explicit_role_request(original) or not is_user_voice_writing(original):
        return None
    return USER_VOICE_ROLE


def should_omit_role(original: str, seeded: Mapping[str, str], role_slot: str | None) -> bool:
    """True when Role was not assigned by the user (labels or act as / you are)."""
    if not role_slot:
        return True
    if role_slot in seeded:
        return False
    return not is_explicit_role_request(original)


def request_already_has_affect(text: str) -> bool:
    """True when the ask already states emotion (don't add a second tone)."""
    return bool(AFFECT_ANY_RE.search(text))


def stated_tone_clause(text: str) -> str | None:
    """Lead-in tone sentence from affect words in the ask, if any."""
    if AFFECT_DISAPPOINTED_RE.search(text):
        return "Use a disappointed tone"
    if AFFECT_PISSED_RE.search(text) or AFFECT_FRUSTRATED_RE.search(text):
        return "Use a frustrated, direct, and firm tone"
    if AFFECT_ANGRY_RE.search(text):
        return "Use an angry, direct, and firm tone"
    return None


def normalize_slot_text(value: str) -> str:
    """Lowercase and strip punctuation for few-shot leak checks."""
    return re.sub(r"\W+", " ", value.lower()).strip()


def matches_fewshot_ban(value: str, banned: frozenset[str]) -> bool:
    """True when a predicted slot copies a system-prompt example fill."""
    if not value or not banned:
        return False
    normalized = normalize_slot_text(value)
    if not normalized:
        return False
    for ban in banned:
        if ban and (ban in normalized or normalized in ban):
            return True
    return False


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
    """`You are a {role}.` Empty value omits the clause."""
    cleaned = tidy(value).rstrip(".")
    cleaned = re.sub(r"(?i)^(you are|act as)\s+", "", cleaned)
    if not cleaned:
        return ""
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
