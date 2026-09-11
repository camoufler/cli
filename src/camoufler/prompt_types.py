"""Heuristic prompt-type detection with optional model tie-break."""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from camoufler.slots import ACTION_RE, is_utterance

logger = logging.getLogger("camoufler")

TIE_DELTA = 1

_ASK_RE = re.compile(
    r"(?i)\b(?:"
    r"how do i|how to|how can i|what is|what's|what are|please |"
    r"can you|could you|act as|walk me through|tell me"
    r")\b"
)


class PromptType(str, Enum):
    """User-prompt intent, plus utterance fallback."""

    FACTUAL = "factual"
    INSTRUCTIONAL = "instructional"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    TRANSFORMATION = "transformation"
    ROLEPLAY = "roleplay"
    STRATEGIC = "strategic"
    UTTERANCE = "utterance"


VALID_LABELS = frozenset(item.value for item in PromptType)

# (type, pattern, weight)
_PATTERNS: tuple[tuple[PromptType, re.Pattern[str], int], ...] = (
    (
        PromptType.FACTUAL,
        re.compile(
            r"(?i)\b(?:what is|what's the|what are|difference between|define|"
            r"who is|when was|how many|how much|versus|\bvs\.?\b)\b"
        ),
        3,
    ),
    (
        PromptType.FACTUAL,
        re.compile(r"(?i)\bexplain (?!how\b)"),
        2,
    ),
    (
        PromptType.INSTRUCTIONAL,
        re.compile(
            r"(?i)\b(?:how do i|how to|how can i|walk me through|step by step|"
            r"steps to|tutorial|set\s*up|configure|install|unit test)\b"
        ),
        3,
    ),
    (
        PromptType.CREATIVE,
        re.compile(
            r"(?i)\b(?:draft|brainstorm|headlines?|tagline|slogan|compelling|"
            r"write (?:an? |me )?(?:poem|story|headline|copy|email|letter|"
            r"complaint|resume|blog)|"
            r"(?:i wanna|i want to) write|"
            r"angry email|"
            r"marketing copy|generate .{0,20}copy)\b"
        ),
        3,
    ),
    (
        PromptType.ANALYTICAL,
        re.compile(
            r"(?i)\b(?:debug|traceback|keyerror|exception|root.?cause|"
            r"what went wrong|fix this (?:code|function|bug)|"
            r"modulenotfounderror|corrected code)\b"
        ),
        3,
    ),
    (
        PromptType.TRANSFORMATION,
        re.compile(
            r"(?i)\b(?:summarize|condense|paraphrase|extract|"
            r"executive summary|translate|rewrite this|format this)\b"
        ),
        3,
    ),
    (
        PromptType.ROLEPLAY,
        re.compile(
            r"(?i)\b(?:act as|role-?play|interview me|conducting a|"
            r"you are an interviewer|ask me three|act as .{0,40}support)\b"
        ),
        3,
    ),
    (
        PromptType.STRATEGIC,
        re.compile(
            r"(?i)\b(?:study plan|roadmap|\d+-week|prepare for|"
            r"certification|hours per week|trade-?offs?|schedule)\b"
        ),
        3,
    ),
)

def looks_like_prompt(text: str) -> bool:
    """True when the text is a request/question rather than a narrative."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    if _ASK_RE.search(stripped):
        return True
    if ACTION_RE.search(stripped):
        return True
    return False


def classify_heuristic(text: str) -> tuple[PromptType, dict[PromptType, int], bool]:
    """Score prompt types. Returns (winner, scores, tied)."""
    scores = {item: 0 for item in PromptType if item is not PromptType.UTTERANCE}
    if is_utterance(text) and not looks_like_prompt(text):
        return PromptType.UTTERANCE, scores, False

    for ptype, pattern, weight in _PATTERNS:
        if pattern.search(text):
            scores[ptype] += weight

    if max(scores.values()) == 0:
        if looks_like_prompt(text):
            if "?" in text or re.search(r"(?i)\b(?:what|why|who|when|where)\b", text):
                scores[PromptType.FACTUAL] = 1
            else:
                scores[PromptType.CREATIVE] = 1
        else:
            return PromptType.UTTERANCE, scores, False

    ranked = sorted(scores, key=lambda item: scores[item], reverse=True)
    top, second = ranked[0], ranked[1]
    tied = scores[second] > 0 and (scores[top] - scores[second]) <= TIE_DELTA
    return top, scores, tied


def parse_label(raw: str) -> PromptType | None:
    """Parse a model classification label; None if invalid."""
    token = raw.strip().split()[0].lower().strip(".,:;") if raw.strip() else ""
    token = token.replace("-", "").replace("_", "")
    aliases = {
        "roleplaying": PromptType.ROLEPLAY.value,
        "roleplay": PromptType.ROLEPLAY.value,
        "howto": PromptType.INSTRUCTIONAL.value,
        "how-to": PromptType.INSTRUCTIONAL.value,
        "info": PromptType.FACTUAL.value,
    }
    token = aliases.get(token, token)
    if token in VALID_LABELS:
        return PromptType(token)
    return None


def classify(
    text: str,
    *,
    model: str | None = None,
    options: dict[str, Any] | None = None,
    classify_chat: Any = None,
) -> PromptType:
    """Heuristic classification; optional model call only on a close tie."""
    winner, scores, tied = classify_heuristic(text)
    if not tied or model is None or classify_chat is None:
        logger.debug("Prompt type %s (heuristic; tied=%s scores=%s)", winner.value, tied, scores)
        return winner
    try:
        label = classify_chat(model, text, options or {})
    except Exception as exc:  # noqa: BLE001 — fall back to heuristic
        logger.info("Classify chat failed (%s); using heuristic %s", exc, winner.value)
        return winner
    parsed = parse_label(label)
    if parsed is None:
        logger.info("Classify chat returned %r; using heuristic %s", label, winner.value)
        return winner
    logger.debug("Prompt type %s (model tie-break; heuristic was %s)", parsed.value, winner.value)
    return parsed
