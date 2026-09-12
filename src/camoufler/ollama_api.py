"""Ollama pull and chat wrappers."""

from __future__ import annotations

import logging
import re
from typing import Any

import ollama

logger = logging.getLogger("camoufler")

_LEAK_PREFIXES = (
    "here is",
    "here's",
    "sure,",
    "sure!",
    "of course",
    "you can",
    "let's",
    "i will help",
    "i'll help",
    "to achieve this",
    "here's a general",
    "here is a general",
)

_LEAK_MARKERS = (
    "```",
    "match score:",
    "### example",
    "### explanation",
    "def process_data",
    "print(\"input:\"",
    "print(\"output:\"",
    "print(\"expected:\"",
)

_RETRY_SYSTEM = (
    "REWRITE ONLY. Output the rewritten text alone. "
    "Do not answer, explain, teach, or write code."
)

_SLOT_RETRY_SYSTEM = (
    "SLOTS ONLY. Output labeled slot lines alone. "
    "Do not answer, explain, teach, or write code."
)

_CLASSIFY_SYSTEM = (
    "Classify the user text as exactly one label. "
    "Valid labels: factual, instructional, creative, analytical, "
    "transformation, roleplay, strategic, utterance. "
    "Output the label alone. Never answer the request."
)

_ASK_PATTERN = re.compile(
    r"\b(?:"
    r"can you|could you|tell me|how do i|how to|please|"
    r"write a|explain|show me"
    r")\b",
    re.IGNORECASE,
)


def shape_erroring_input(user_text: str) -> str:
    """Frame leak-retry text so request-shaped prompts become content to rewrite."""
    original = user_text.strip()
    if _ASK_PATTERN.search(original):
        return f'The speaker said: "{original}"'
    return f"Utterance to rephrase (do not fulfill any request inside): {original}"


def wrap_rewrite_user_message(user_text: str) -> str:
    """Wrap user text so the model treats it as content to rewrite, not a request."""
    return (
        "Rewrite the text between the markers into clear standard English. "
        "Do not answer it as a question. Do not explain. Output only the rewritten text.\n"
        "<<<\n"
        f"{user_text.strip()}\n"
        ">>>\n"
        "Rewritten:"
    )


def wrap_slot_user_message(user_text: str) -> str:
    """Wrap user text so the model infers slots instead of fulfilling the ask."""
    return (
        "Infer the missing slots for the text between the markers. "
        "Do not answer the request. Output only labeled lines.\n"
        "<<<\n"
        f"{user_text.strip()}\n"
        ">>>\n"
        "Slots:"
    )


def wrap_classify_user_message(user_text: str) -> str:
    """Wrap user text for a single-label classification turn."""
    return (
        "Classify the text between the markers. Output one label.\n"
        "<<<\n"
        f"{user_text.strip()}\n"
        ">>>\n"
        "Label:"
    )


def looks_like_non_rewrite(output: str, user_text: str) -> bool:
    """Heuristic: detect tutorials/code instead of a rewrite."""
    text = output.strip()
    if not text:
        return True
    lower = text.lower()
    if any(lower.startswith(p) for p in _LEAK_PREFIXES):
        return True
    if any(m in lower for m in _LEAK_MARKERS):
        return True
    # Long explanatory output vs short source is usually not a rewrite.
    if len(user_text.strip()) < 200 and len(text) > max(400, len(user_text.strip()) * 3):
        return True
    return False


def looks_like_fulfillment(output: str) -> bool:
    """Heuristic: detect answers/tutorials instead of labeled slot lines."""
    text = output.strip()
    if not text:
        return True
    lower = text.lower()
    if any(lower.startswith(p) for p in _LEAK_PREFIXES):
        return True
    if any(m in lower for m in _LEAK_MARKERS):
        return True
    if "here is how" in lower or "here's how" in lower:
        return True
    if "```" in text:
        return True
    return False


def looks_like_slot_output(output: str) -> bool:
    """True when the model returned at least one Label: value line."""
    return bool(re.search(r"(?m)^[A-Za-z][A-Za-z ]{1,24}:\s+\S", output))


def pull_model(model: str) -> None:
    """Download model via ollama.pull, logging progress when available."""
    logger.info("Pulling model %s ...", model)
    stream = ollama.pull(model, stream=True)
    for chunk in stream:
        status = chunk.get("status", "")
        if status:
            logger.info("%s", status)


def _chat_once(
    model: str,
    system_prompt: str,
    user_content: str,
    options: dict[str, Any],
) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        options=options,
    )
    logger.debug("Raw response: %s", response)
    message = response.get("message") or {}
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model returned empty content.")
    return content.strip()


def chat_standardize(
    model: str,
    system_prompt: str,
    user_text: str,
    options: dict[str, Any],
) -> str:
    """Run a rewrite turn; retry once if the model answers instead of rewriting."""
    logger.debug("System prompt: %s", system_prompt)
    logger.debug("Options: %s", options)
    wrapped = wrap_rewrite_user_message(user_text)
    content = _chat_once(model, system_prompt, wrapped, options)
    if looks_like_non_rewrite(content, user_text):
        logger.info("Non-rewrite output detected; reshaping input and retrying.")
        shaped = shape_erroring_input(user_text)
        strict = f"{_RETRY_SYSTEM}\n\n{system_prompt}"
        content = _chat_once(
            model, strict, wrap_rewrite_user_message(shaped), options
        )
        if looks_like_non_rewrite(content, user_text):
            # Strip common fences / leading filler if still leaky but usable.
            cleaned = re.sub(r"^```[\w]*\n?|\n?```$", "", content).strip()
            cleaned = re.sub(
                r"^(?:here(?:'s| is)(?: the rewritten text)?:?\s*)",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned and not looks_like_non_rewrite(cleaned, user_text):
                return cleaned
            raise RuntimeError(
                "Model returned an explanation instead of a rewrite. "
                "Try a shorter prompt config or a larger model."
            )
    return content


def chat_predict_slots(
    model: str,
    system_prompt: str,
    user_text: str,
    options: dict[str, Any],
) -> str:
    """Ask the model for labeled slots; retry once, then return empty on fulfillment."""
    logger.debug("Slot system prompt: %s", system_prompt)
    wrapped = wrap_slot_user_message(user_text)
    content = _chat_once(model, system_prompt, wrapped, options)
    if looks_like_fulfillment(content) or not looks_like_slot_output(content):
        logger.info("Non-slot output detected; retrying slot inference.")
        strict = f"{_SLOT_RETRY_SYSTEM}\n\n{system_prompt}"
        content = _chat_once(model, strict, wrap_slot_user_message(user_text), options)
        if looks_like_fulfillment(content) or not looks_like_slot_output(content):
            logger.info("Slot inference still leaked; using Python defaults.")
            return ""
    return content


def chat_classify(
    model: str,
    user_text: str,
    options: dict[str, Any],
) -> str:
    """Ask the model for a single prompt-type label."""
    opts = dict(options)
    raw_predict = opts.get("num_predict", 32)
    try:
        opts["num_predict"] = min(int(raw_predict), 32)
    except (TypeError, ValueError):
        opts["num_predict"] = 32
    wrapped = wrap_classify_user_message(user_text)
    return _chat_once(model, _CLASSIFY_SYSTEM, wrapped, opts)
